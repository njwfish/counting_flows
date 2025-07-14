import torch
import torch.nn as nn
import torch.nn.functional as F

class ARDigitTransformer(nn.Module):
    def __init__(self, L: int, base: int, x_dim: int, context_dim: int = 0,
                 d_model=64, nhead=4, num_layers=2, hidden=128):
        super().__init__()
        self.L = L  # max digits per integer
        self.base = base
        self.D = x_dim  # number of integers per list
        self.sos_token = base  # use index 'base' for SOS
        self.sign_token = base + 1  # use index 'base+1' for negative sign

        self._in_dim = x_dim * 2 + 1 + context_dim 
        
        # Shared MLP for input processing (similar to energy model)
        # Takes all D integers and produces shared context
        self.shared_mlp = nn.Sequential(
            nn.Linear(self._in_dim, hidden),
            nn.SELU(),
            nn.Linear(hidden, hidden), 
            nn.SELU(),
            nn.Linear(hidden, hidden),
            nn.SELU(),
            nn.Linear(hidden, d_model),
        )
        
        # Position embeddings for each integer position in the list
        self.position_embed = nn.Embedding(x_dim, d_model)
        
        # embed previous output‐digits (0..base-1 plus SOS plus sign)
        self.token_embed = nn.Embedding(base + 2, d_model)
        # positional embeddings for digit positions
        self.pos_embed = nn.Embedding(L, d_model)
        
        # transformer encoder as a causal decoder (with 2*d_model input)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=2*d_model, nhead=nhead, batch_first=False, dim_feedforward=2 * hidden
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        # project to digit logits (0..base-1 plus SOS plus sign)
        self.out_proj = nn.Linear(2*d_model, base + 2)

    def _int_to_digit_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert integers to digit sequences (LSB first) with sign handling
        x: (B, D) integers
        returns: (B, D, L) digit sequence with sign token if negative
        """
        B, D = x.size()
        device = x.device
        
        # Handle signs
        is_negative = x < 0
        abs_x = torch.abs(x)
        
        # Convert to digits (LSB first) - fully vectorized
        digits = torch.zeros(B, D, self.L, dtype=torch.long, device=device)
        
        for i in range(self.L):
            digits[:, :, i] = abs_x % self.base
            abs_x = abs_x // self.base
            
        # Add sign token for negative numbers - vectorized
        # Find number of actual digits for each integer
        abs_vals = torch.abs(x)
        # Count digits by finding when abs_vals becomes 0 after repeated division
        temp_vals = abs_vals.clone()
        digit_counts = torch.ones_like(abs_vals)
        for i in range(self.L - 1):
            temp_vals = temp_vals // self.base
            digit_counts += (temp_vals > 0).long()
        
        # Place sign token after last digit for negative numbers
        digit_counts = torch.clamp(digit_counts, max=self.L-1)  # Ensure we don't exceed bounds
        batch_idx, list_idx = torch.where(is_negative)
        if len(batch_idx) > 0:
            sign_positions = digit_counts[batch_idx, list_idx]
            valid_positions = sign_positions < self.L
            if valid_positions.any():
                valid_batch_idx = batch_idx[valid_positions]
                valid_list_idx = list_idx[valid_positions] 
                valid_sign_pos = sign_positions[valid_positions]
                digits[valid_batch_idx, valid_list_idx, valid_sign_pos] = self.sign_token
                    
        return digits

    def _digit_sequence_to_int(self, digits: torch.Tensor) -> torch.Tensor:
        """
        Convert digit sequences back to integers - vectorized
        digits: (B, D, L) digit sequences 
        returns: (B, D) integers
        """
        B, D, L = digits.size()
        device = digits.device
        
        # Find sign positions
        sign_mask = (digits == self.sign_token)
        has_sign = sign_mask.any(dim=2)  # (B, D)
        
        # Create base powers: [1, base, base^2, ...]
        base_powers = self.base ** torch.arange(L, device=device)  # (L,)
        
        # Mask out sign tokens and invalid digits
        valid_digit_mask = (digits >= 0) & (digits < self.base)
        masked_digits = torch.where(valid_digit_mask, digits, 0)
        
        # Compute values using vectorized multiplication
        values = (masked_digits * base_powers.unsqueeze(0).unsqueeze(0)).sum(dim=2)  # (B, D)
        
        # Apply sign
        result = torch.where(has_sign, -values, values)
        
        return result

    def forward(self, x: torch.IntTensor, prev_digits: torch.IntTensor, M_t, t, z=None) -> torch.Tensor:
        """
        x: (B, D) raw integer inputs (can be negative) - list of D integers per batch
        prev_digits: (B, D, L) ints in {0..base-1, SOS=base, sign=base+1}, LSB-first shifted right
        returns: (B, D, L, base+2) logits over next digit/sign at each position for each integer
        """
        B, D, L = prev_digits.size()
        device = x.device

        # Generate shared context from all input integers using MLP
        x_float = x.float()  # (B, D)
        M_t = M_t.float()    # (B, D)
        t = t.float()        # (B, 1)
        if z is not None:
            z = z.float()
            inp = torch.cat([x_float, M_t, t, z], dim=-1)
        else:
            inp = torch.cat([x_float, M_t, t], dim=-1)
        shared_context = self.shared_mlp(inp)  # (B, d_model)
        
        # Get position embeddings for each integer position
        pos_ids = torch.arange(D, device=device)  # [0, 1, 2, ..., D-1]
        pos_emb = self.position_embed(pos_ids)  # (D, d_model)
        
        # Combine shared context with position embeddings for each position via concatenation
        # shared_context: (B, d_model) -> (B, 1, d_model) -> (B, D, d_model)
        # pos_emb: (D, d_model) -> (1, D, d_model) -> (B, D, d_model)
        shared_expanded = shared_context.unsqueeze(1).expand(B, D, -1)  # (B, D, d_model)
        pos_expanded = pos_emb.unsqueeze(0).expand(B, D, -1)  # (B, D, d_model)
        combined_input = torch.cat([shared_expanded, pos_expanded], dim=-1)  # (B, D, 2*d_model)

        # Process each integer position independently
        # Reshape to treat each position as a separate batch item
        combined_flat = combined_input.reshape(B * D, -1)  # (B*D, 2*d_model)
        prev_digits_flat = prev_digits.reshape(B * D, L)  # (B*D, L)
        
        # Token embeddings for previous digits
        tok_emb = self.token_embed(prev_digits_flat)  # (B*D, L, d_model)
        
        # Digit positional embeddings
        digit_pos_ids = torch.arange(L, device=device)
        digit_pos_emb = self.pos_embed(digit_pos_ids)  # (L, d_model)
        
        # Build sequence for transformer: each integer processed independently
        # Expand combined input to all digit positions
        inp_expanded = combined_flat.unsqueeze(1).expand(B*D, L, -1)  # (B*D, L, 2*d_model)
        
        # Pad token and positional embeddings to match 2*d_model
        pad_size = inp_expanded.size(-1) - tok_emb.size(-1)  # d_model
        tok_emb_padded = F.pad(tok_emb, (0, pad_size))  # (B*D, L, 2*d_model)
        
        digit_pos_expanded = digit_pos_emb.unsqueeze(0).expand(B*D, L, -1)  # (B*D, L, d_model)
        digit_pos_padded = F.pad(digit_pos_expanded, (0, pad_size))  # (B*D, L, 2*d_model)
        
        # Combine embeddings
        src = inp_expanded + tok_emb_padded + digit_pos_padded  # (B*D, L, 2*d_model)
        
        # Reshape for transformer: (L, B*D, 2*d_model)
        src = src.transpose(0, 1)  # (L, B*D, 2*d_model)
        
        # Simple causal mask - each position can only attend to previous positions of the SAME integer
        causal_mask = torch.triu(torch.full((L, L), float('-inf'), device=device), diagonal=1)
        
        # Process all integers in parallel
        out = self.transformer(src, mask=causal_mask)  # (L, B*D, 2*d_model)
        out = out.transpose(0, 1)  # (B*D, L, 2*d_model)
        
        # Project to logits
        logits = self.out_proj(out)  # (B*D, L, base+2)
        
        # Reshape back to original batch structure
        logits = logits.reshape(B, D, L, self.base + 2)  # (B, D, L, base+2)
        
        return logits

    def loss(self, target_ints: torch.Tensor, x, M_t, t, z=None) -> torch.Tensor:
        """
        Compute autoregressive loss for target integer lists
        x: (B, D) input integer lists 
        target_ints: (B, D) target integer lists to predict
        returns: scalar loss
        """
        B, D = x.size()
        device = x.device

        target_ints = (target_ints - x).to(torch.int32)
        
        # Convert target integers to digit sequences
        target_digits = self._int_to_digit_sequence(target_ints)  # (B, D, L)
        
        # Create shifted input for autoregressive training
        # Start with SOS token, then shift target digits right
        prev_digits = torch.full((B, D, self.L), self.sos_token, dtype=torch.long, device=device)
        prev_digits[:, :, 1:] = target_digits[:, :, :-1]
        
        # Forward pass
        logits = self.forward(x, prev_digits, M_t, t, z)  # (B, D, L, base+2)
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(
            logits.reshape(-1, self.base + 2),  # (B*D*L, base+2)
            target_digits.reshape(-1),          # (B*D*L,)
            ignore_index=-1  # ignore padding if we add it later
        )
        
        return loss

    def sample(self, x: torch.Tensor, M_t, t, z=None, temperature: float = 1.0) -> torch.Tensor:
        """
        Sample digit sequences autoregressively for each integer in parallel
        x: (B, D) input integer lists
        temperature: sampling temperature (1.0 = original logits)
        returns: (B, D) sampled integer lists
        """
        B, D = x.size()
        device = x.device
        
        # Initialize with SOS tokens
        prev_digits = torch.full((B, D, self.L), self.sos_token, dtype=torch.long, device=device)
        
        self.eval()
        with torch.no_grad():
            # Process all positions sequentially, but all integers in parallel
            for pos in range(self.L):
                # Get logits for current position across all integers
                logits = self.forward(x, prev_digits, M_t, t, z)  # (B, D, L, base+2)
                current_logits = logits[:, :, pos, :] / temperature  # (B, D, base+2)
                
                # Sample from logits for all integers in parallel
                probs = F.softmax(current_logits, dim=-1)  # (B, D, base+2)
                # Reshape to sample all at once
                probs_flat = probs.reshape(B*D, self.base + 2)
                sampled_flat = torch.multinomial(probs_flat, num_samples=1).squeeze(-1)  # (B*D,)
                sampled = sampled_flat.reshape(B, D)  # (B, D)
                
                # Update prev_digits for next position if not at last position
                if pos < self.L - 1:
                    prev_digits[:, :, pos + 1] = sampled
        
        # Convert sampled digit sequences back to integers
        # Get final sampled digits by running one more forward pass
        final_logits = self.forward(x, prev_digits, M_t, t, z)  # (B, D, L, base+2)
        final_sampled = torch.argmax(final_logits, dim=-1)  # (B, D, L)
        
        return self._digit_sequence_to_int(final_sampled) + x

    