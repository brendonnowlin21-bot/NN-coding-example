class DynamicInputScaler(nn.Module):

    ENG_START = 25     # First engineered feature index (after 25 raw SMC features)
    ACCT_START = 41    # First account feature index (pass-through, already 0-1)
    TOTAL_FEATS = 57   # Total feature count
    EMA_FEATS = ACCT_START - ENG_START  # 16 engineered features to z-score (25-40)
    # OLD_EMA_FEATS = 33 removed (2026-05-16): was kept for checkpoint compat with
    # a pre-refactor architecture that no longer exists. Checkpoints load with
    # strict=False + shape filtering, so mismatched buffers are re-initialized.
    
    def __init__(self, momentum: float = 0.01, eps: float = 1e-5):
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        self.register_buffer('running_mean', torch.zeros(self.EMA_FEATS))
        self.register_buffer('running_var', torch.ones(self.EMA_FEATS))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        self._needs_ema_reset = False
        self.warmup_batches = 10
    
    def reset_ema_stats(self):
        """Reset EMA running statistics â€” call after loading from corrupted checkpoint."""
        with torch.no_grad():
            self.running_mean.zero_()
            self.running_var.fill_(1.0)
            self.num_batches_tracked.zero_()
        self._needs_ema_reset = False
    def forward(self, x):
        # x: (batch, seq_len, 57)
        if self._needs_ema_reset:
            self.reset_ema_stats()

        # SILENT-CORRUPTION GUARD: track NaN/Inf inputs. Any upstream bug that
        # produces non-finite values would otherwise be masked by nan_to_num
        # below (NaN â†’ 0.0 = "neutral feature"). Counting and periodic-logging
        # surfaces the failure rate. Zero cost on healthy inputs.
        if not hasattr(self, '_nan_input_count'):
            self._nan_input_count = 0
            self._nan_input_logged = 0
        _finite_mask = torch.isfinite(x)
        if not _finite_mask.all():
            self._nan_input_count += int((~_finite_mask).sum().item())
            # Log once per doubling of the count so noise stays bounded.
            if self._nan_input_count >= max(1, 2 * self._nan_input_logged):
                logging.warning(
                    f"[DynamicInputScaler] non-finite input encountered "
                    f"(cumulative count = {self._nan_input_count}). Silently replaced with 0.0."
                )
                self._nan_input_logged = self._nan_input_count

        x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        x = torch.clamp(x, min=-1e9, max=1e9)
        
        open_prices = x[:, :, 0]
        mask = (open_prices.abs() > 1e-6).long()
        first_valid_idx = torch.argmax(mask, dim=1)
        ref_price = open_prices.gather(1, first_valid_idx.unsqueeze(1)).unsqueeze(2)
        ref_price = torch.clamp(ref_price.abs(), min=1.0)  # Minimum 1.0 for random inputs
        
        # Build output tensor piece by piece (no in-place modification)
        # OHLC features (0:4) - normalize by reference price
        ohlc = ((x[:, :, 0:4] / ref_price) - 1.0) * 100.0
        ohlc = torch.clamp(ohlc, min=-1000.0, max=1000.0)
        
        # Volume feature (4) - log transform
        volume = torch.log1p(x[:, :, 4:5].abs()) / 10.0
        
        # Features 5:12 - pass through unchanged (binary SMC signals, now truly 0/1)
        mid_features = x[:, :, 5:12]
        
        # SMC level features (12:19) - normalize by reference price
        smc_levels = ((x[:, :, 12:19] / ref_price) - 1.0) * 100.0
        smc_levels = torch.clamp(smc_levels, min=-1000.0, max=1000.0)
        
        # Features 19:23 - pass through (SMC index distances, now normalized 0-1 by precompute)
        pre_vol_features = x[:, :, 19:23]
        # OBVolume feature (23) - log transform
        ob_volume = torch.log1p(x[:, :, 23:24].abs()) / 10.0
        # Percentage is a raw SMC feature (ALL_RAW_FEATURES index 24), not engineered.
        percentage = x[:, :, 24:25] / 100.0
        percentage = torch.clamp(percentage, min=0.0, max=1.0)
        
        # === Engineered features (25:41) â€” z-score normalize ===
        # Account features 41-56 are ALREADY 0-1 normalized by mode_handler.
        eng_raw = x[:, :, self.ENG_START:self.ACCT_START]  # (batch, seq_len, 16)
        acct_raw = x[:, :, self.ACCT_START:]  # (batch, seq_len, 16) â€” pass-through
        
        # Update running statistics during training (EMA approach)
        if self.training:
            with torch.no_grad():
                feat_mean = eng_raw.mean(dim=(0, 1))  # (16,)
                feat_var = eng_raw.var(dim=(0, 1), unbiased=False)  # (16,)
                # FIX (BUG #14): Clamp variance to prevent near-zero values
                feat_var = torch.clamp(feat_var, min=1e-4)
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * feat_mean
                self.running_var  = (1 - self.momentum) * self.running_var  + self.momentum * feat_var
                self.num_batches_tracked += 1
        
        # Apply z-score normalization to engineered features once warmed up
        if self.num_batches_tracked >= self.warmup_batches:
            mean = self.running_mean.unsqueeze(0).unsqueeze(0)
            std = torch.sqrt(torch.clamp(self.running_var, min=1e-4).unsqueeze(0).unsqueeze(0) + self.eps)
            eng_normalized = torch.clamp((eng_raw - mean) / std, min=-10.0, max=10.0)
        else:
            eng_normalized = torch.clamp(eng_raw, min=-100.0, max=100.0)
        
        # Account features: pass-through with light clamping (already 0-1 normalized)
        acct_clamped = torch.clamp(acct_raw, min=-2.0, max=2.0)
        
        # Concatenate all parts (total: 4+1+7+7+4+1+1+16+16 = 57)
        out = torch.cat([
            ohlc,              # 0:4   (4 features)
            volume,            # 4:5   (1 feature)
            mid_features,      # 5:12  (7 features â€” binary SMC signals)
            smc_levels,        # 12:19 (7 features â€” ref-price normalized)
            pre_vol_features,  # 19:23 (4 features â€” normalized distances)
            ob_volume,         # 23:24 (1 feature â€” log transformed)
            percentage,        # 24:25 (1 feature â€” /100 normalized)
            eng_normalized,    # 25:41 (16 features â€” z-score normalized)
            acct_clamped       # 41:57 (16 features â€” pass-through, already 0-1)
        ], dim=-1)
        
        out = torch.nan_to_num(out, nan=0.0, posinf=100.0, neginf=-100.0)
        return out






#actual NN
class HybridLayer(nn.Module):

    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        
        # === SHARED REPRESENTATION SPACE ===
        self.pre_parallel_norm = nn.LayerNorm(d_model)  # before CNN/LSTM/Attention branch
        self.post_fusion_norm = nn.LayerNorm(d_model)   # after fused output, pre-residual-add
        self.pre_ff_norm = nn.LayerNorm(d_model)        # before feedforward sub-block
        
        # === CNN MECHANISM (Local Pattern Detection) ===
        self.conv_small = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=conv_groups)
        self.conv_medium = nn.Conv1d(d_model, d_model, kernel_size=5, padding=2, groups=conv_groups)
        self.conv_large = nn.Conv1d(d_model, d_model, kernel_size=7, padding=3, groups=conv_groups)
        self.conv_small_norm = nn.GroupNorm(conv_groups, d_model)
        self.conv_medium_norm = nn.GroupNorm(conv_groups, d_model)
        self.conv_large_norm = nn.GroupNorm(conv_groups, d_model)
        self.conv_act = nn.GELU()
        self.conv_proj = nn.Linear(d_model * 3, d_model)
        
        # === LSTM MECHANISM (Sequential Memory) ===
        self.lstm = nn.LSTM(d_model, d_model // 2, num_layers=1, bidirectional=True, batch_first=True)
        
        # === LSTM INIT: Orthogonal Weights + Moderate Gate Bias ===
        hidden_size = d_model // 2  # 128 for d_model=256
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name or 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                param.data.zero_()
                param.data[0:hidden_size].fill_(0.0)                # Input gate: balanced (sigmoid=0.50)
                param.data[hidden_size:2*hidden_size].fill_(0.5)    # Forget gate: moderate retention (sigmoid=0.73)
                # cell_gate (2*hs:3*hs) stays at 0.0 (neutral)
                param.data[3*hidden_size:4*hidden_size].fill_(0.25) # Output gate: slightly open (sigmoid=0.62)
        
        # === ATTENTION MECHANISM (Long-Range Dependencies) ===
        self.attention = nn.MultiheadAttention(d_model, num_heads=num_heads, dropout=dropout, batch_first=True)
        
        # === ATTENTION INIT: Orthogonal Q/K + Xavier V (prevents Q/K death spiral) ===
        with torch.no_grad():
            d = d_model
            if self.attention.in_proj_weight is not None:
                # in_proj_weight: (3*d_model, d_model) = [Q; K; V] stacked
                q_weight = torch.empty(d, d)
                k_weight = torch.empty(d, d)
                v_weight = torch.empty(d, d)
                nn.init.orthogonal_(q_weight)    # Preserves dot-product geometry
                nn.init.orthogonal_(k_weight)    # Matching angular structure
                nn.init.xavier_normal_(v_weight) # Diverse value projections
                self.attention.in_proj_weight[:d].copy_(q_weight)
                self.attention.in_proj_weight[d:2*d].copy_(k_weight)
                self.attention.in_proj_weight[2*d:].copy_(v_weight)
            if self.attention.in_proj_bias is not None:
                nn.init.constant_(self.attention.in_proj_bias, 0.0)
        
        # === ALiBi: Per-Head Relative Position Bias (Press et al., 2022) ===
        num_h = num_heads
        alibi_slopes = 2.0 ** (-torch.arange(1, num_h + 1, dtype=torch.float32) * 8.0 / num_h)
        self.register_buffer('alibi_slopes', alibi_slopes)  # (num_heads,)
        self._num_heads = num_heads
        
        # === FUSION: ConcatMLP (Scientifically validated 2026-07-XX) ===
        self.fusion_mlp = nn.Sequential(
            nn.Linear(d_model * 3, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.fusion_proj_cnn = nn.Linear(d_model, d_model, bias=False)
        self.fusion_proj_lstm = nn.Linear(d_model, d_model, bias=False)
        self.fusion_proj_trans = nn.Linear(d_model, d_model, bias=False)
        self.pathway_norm_cnn = nn.LayerNorm(d_model)
        self.pathway_norm_lstm = nn.LayerNorm(d_model)
        self.pathway_norm_trans = nn.LayerNorm(d_model)
        self.fusion_mix_alpha = nn.Parameter(torch.tensor(0.0))
        self.fusion_residual_logits = nn.Parameter(torch.tensor([0.1, 0.0, -0.1]))
        self.fusion_residual_temp = 0.3
        
        # === FEEDFORWARD (Shared Processing) ===
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def verify_architecture_health(self) -> dict:
        results = {}
        
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                hidden_size = self.d_model // 2
                forget_bias_mean = param.data[hidden_size:2*hidden_size].mean().item()
                ok = forget_bias_mean >= 0.3
                results[f'lstm_forget_bias ({name})'] = ok
                if not ok:
                    logging.warning(
                        f"[ARCH HEALTH] LSTM forget gate bias too low: {forget_bias_mean:.3f} "
                        f"(should be >= 0.3, currently â†’ sigmoid = {1/(1+math.exp(-forget_bias_mean)):.3f} retention)"
                    )
        
        if self.attention.in_proj_weight is not None:
            d = self.d_model
            q_norm = self.attention.in_proj_weight[:d].norm().item()
            k_norm = self.attention.in_proj_weight[d:2*d].norm().item()
            v_norm = self.attention.in_proj_weight[2*d:].norm().item()
            ok = q_norm > 0.1 and k_norm > 0.1
            results['attention_qk_alive'] = ok
            if not ok:
                logging.warning(
                    f"[ARCH HEALTH] Attention Q/K weights near zero: "
                    f"Q_norm={q_norm:.4f}, K_norm={k_norm:.4f}, V_norm={v_norm:.4f}"
                )
            else:
                logging.info(
                    f"[ARCH HEALTH] Attention norms OK: Q={q_norm:.2f}, K={k_norm:.2f}, V={v_norm:.2f}"
                )
        
        for i, layer in enumerate(self.fusion_mlp):
            if hasattr(layer, 'weight'):
                w_norm = layer.weight.data.norm().item()
                ok = 0.01 < w_norm < 200.0
                results[f'fusion_mlp_layer{i}_weight_norm'] = ok
                if not ok:
                    logging.warning(f"[ARCH HEALTH] Fusion MLP layer {i} weight norm out of range: {w_norm:.4f}")
        
        # CHECK 4: No NaN in any parameter
        nan_found = False
        for name, param in self.named_parameters():
            if torch.isnan(param.data).any():
                nan_found = True
                logging.error(f"[ARCH HEALTH] NaN found in parameter: {name}")
        results['no_nans'] = not nan_found
        
        # CHECK 5: Fusion mix alpha in reasonable range
        alpha = torch.sigmoid(self.fusion_mix_alpha).item()
        ok = 0.05 < alpha < 0.95
        results['fusion_mix_alpha'] = ok
        if not ok:
            logging.warning(f"[ARCH HEALTH] Fusion mix alpha extreme: {alpha:.4f} (should be 0.05-0.95)")
        
        # CHECK 6: Fusion projection norm balance (detects CNN collapse)
        proj_norms = {
            'cnn': self.fusion_proj_cnn.weight.data.norm().item(),
            'lstm': self.fusion_proj_lstm.weight.data.norm().item(),
            'trans': self.fusion_proj_trans.weight.data.norm().item(),
        }
        max_norm = max(proj_norms.values())
        min_norm = min(proj_norms.values())
        ratio = max_norm / (min_norm + 1e-8)
        ok = ratio < 2.0  # All projections within 2x of each other
        results['fusion_proj_balanced'] = ok
        proj_str = ", ".join(f"{k}={v:.3f}" for k, v in proj_norms.items())
        if not ok:
            logging.warning(f"[ARCH HEALTH] Fusion projection norms imbalanced (ratio={ratio:.2f}): {proj_str}")
        else:
            logging.info(f"[ARCH HEALTH] Fusion projection norms balanced: {proj_str} (ratio={ratio:.2f})")
        
        # Summary
        all_passed = all(results.values())
        if all_passed:
            logging.info("[ARCH HEALTH] All checks PASSED âœ“")
        else:
            failed = [k for k, v in results.items() if not v]
            logging.warning(f"[ARCH HEALTH] FAILED checks: {failed}")
        
        return results
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        residual = x
        
        # === PARALLEL MECHANISMS ON SHARED REPRESENTATION ===
        x_norm = self.pre_parallel_norm(x)
        
        # 1. CNN: Local pattern extraction (multi-scale)
        x_conv = x_norm.transpose(1, 2)  # (batch, d_model, seq_len)
        conv_small = self.conv_act(self.conv_small_norm(self.conv_small(x_conv)))
        conv_medium = self.conv_act(self.conv_medium_norm(self.conv_medium(x_conv)))
        conv_large = self.conv_act(self.conv_large_norm(self.conv_large(x_conv)))
        x_conv = torch.cat([conv_small, conv_medium, conv_large], dim=1).transpose(1, 2)  # (batch, seq_len, d_model*3)
        x_conv = self.conv_proj(x_conv)  # (batch, seq_len, d_model)
        
        if self.training and seq_len > 30:
            _tbptt_chunks = []
            _tbptt_hc = None
            for _i in range(0, seq_len, 30):
                _chunk = x_norm[:, _i:_i+30, :]
                if _tbptt_hc is not None:
                    _tbptt_hc = (_tbptt_hc[0].detach(), _tbptt_hc[1].detach())
                _chunk_out, _tbptt_hc = self.lstm(_chunk, _tbptt_hc)
                _tbptt_chunks.append(_chunk_out)
            x_lstm = torch.cat(_tbptt_chunks, dim=1)  # (batch, seq_len, d_model)
        else:
            x_lstm, _ = self.lstm(x_norm)  # (batch, seq_len, d_model)
        
        seq_len_attn = x_norm.size(1)
        cache_key = (seq_len_attn, x_norm.device, x_norm.dtype)
        rel_dist = self._alibi_cache.get(cache_key) if hasattr(self, '_alibi_cache') else None
        if rel_dist is None:
            if not hasattr(self, '_alibi_cache'):
                self._alibi_cache = {}
            positions = torch.arange(seq_len_attn, device=x_norm.device, dtype=x_norm.dtype)
            rel_dist = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs()  # (S, S)
            self._alibi_cache[cache_key] = rel_dist
        # (H, S, S): head 0 has steep slope (local), head 7 has shallow (global)
        alibi_per_head = -self.alibi_slopes.view(-1, 1, 1) * rel_dist.unsqueeze(0)  # (H, S, S)
        # Batch-major expand: (B, H, S, S) -> (B*H, S, S) â€” PyTorch MHA's expected shape
        alibi_bias = alibi_per_head.unsqueeze(0).expand(batch_size, -1, -1, -1)
        alibi_bias = alibi_bias.reshape(batch_size * self._num_heads, seq_len_attn, seq_len_attn)
        x_attn, _ = self.attention(x_norm, x_norm, x_norm, attn_mask=alibi_bias)  # (batch, seq_len, d_model)
        
        # === ConcatMLP FUSION ===
        x_conv = self.pathway_norm_cnn(x_conv)
        x_lstm = self.pathway_norm_lstm(x_lstm)
        x_attn = self.pathway_norm_trans(x_attn)

        if self.training:
            x_conv, x_lstm, x_attn = PathwayGradEqualizer.apply(x_conv, x_lstm, x_attn)

        concatenated = torch.cat([x_conv, x_lstm, x_attn], dim=-1)  # (B, S, 3*D)
        mlp_out = self.fusion_mlp(concatenated)  # (B, S, D)

        proj_cnn = self.fusion_proj_cnn(x_conv)
        proj_lstm = self.fusion_proj_lstm(x_lstm)
        proj_trans = self.fusion_proj_trans(x_attn)
        residual_w = torch.softmax(self.fusion_residual_logits / self.fusion_residual_temp, dim=0)  # (3,)
        avg_proj = residual_w[0] * proj_cnn + residual_w[1] * proj_lstm + residual_w[2] * proj_trans

        # Mix: alpha * MLP + (1 - alpha) * residual
        alpha = torch.sigmoid(self.fusion_mix_alpha)
        x_fused = alpha * mlp_out + (1 - alpha) * avg_proj
        
        # Store individual pathway representations BEFORE fusion for diagnostics
        self.last_pathway_reps = {
            'cnn': x_conv.detach(),
            'lstm': x_lstm.detach(),
            'transformer': x_attn.detach()
        }
        self.last_pathway_reps_live = {
            'cnn': x_conv.mean(dim=1),       # (batch, d_model)
            'lstm': x_lstm.mean(dim=1),      # full-sequence mean â€” gradient to all timesteps
            'transformer': x_attn.mean(dim=1) # (batch, d_model)
        }
        with torch.no_grad():
            cnn_energy = proj_cnn.detach().norm(dim=-1).mean().item()
            lstm_energy = proj_lstm.detach().norm(dim=-1).mean().item()
            trans_energy = proj_trans.detach().norm(dim=-1).mean().item()
            total_energy = cnn_energy + lstm_energy + trans_energy + 1e-8

            cnn_pct = cnn_energy / total_energy * 100
            lstm_pct = lstm_energy / total_energy * 100
            trans_pct = trans_energy / total_energy * 100
            self.last_contributions = {
                'cnn': cnn_pct,
                'lstm': lstm_pct,
                'transformer': trans_pct,
            }
            self._last_gate_stats = {
                'cnn_dominance_pct': cnn_pct,
                'lstm_dominance_pct': lstm_pct,
                'transformer_dominance_pct': trans_pct,
                'fusion_mix_alpha': alpha.item(),
                'residual_w_cnn': float(residual_w[0].item()),
                'residual_w_lstm': float(residual_w[1].item()),
                'residual_w_trans': float(residual_w[2].item()),
            }
        
        # Post-fusion normalization: stabilizes combined CNN+LSTM+Attention output
        x_fused = self.post_fusion_norm(x_fused)
        
        # Residual connection
        x = residual + self.dropout(x_fused)
        
        # === SHARED FEEDFORWARD ===
        residual = x
        x = self.pre_ff_norm(x)
        x = residual + self.ff(x)
        
        return x, self.last_contributions


#fusion model:
class HybridPerceptionBackbone(nn.Module):
    def __init__(self, input_size: int, d_model: int = 256, num_layers: int = 4, num_heads: int = 8, max_seq_len: int = 200):
        super().__init__()
        
        # === SHARED INPUT EMBEDDING ===
        # All mechanisms start from the same representation
        self.input_embedding = nn.Sequential(
            nn.Linear(input_size, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        self.raw_proj = nn.Linear(input_size, d_model)
        nn.init.normal_(self.raw_proj.weight, mean=0.0, std=0.1)
        nn.init.zeros_(self.raw_proj.bias)
        self.raw_mix_gate = nn.Parameter(torch.tensor(-1.0))
        
        # === SINUSOIDAL POSITIONAL ENCODING ===
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('positional_encoding', pe.unsqueeze(0))
        self.pe_scale = nn.Parameter(torch.tensor(1.0))
        
        # === STACK OF UNIFIED HYBRID LAYERS ===
        self.layers = nn.ModuleList([
            HybridLayer(d_model=d_model, num_heads=num_heads, dropout=0.1)
            for _ in range(num_layers)
        ])
        self.pre_pool_norm = nn.LayerNorm(d_model)
        self.attention_pool = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True, dropout=0.1)
        self.pool_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pool_skip_alpha = nn.Parameter(torch.tensor(2.0))
        
        # === FINAL PROJECTION ===
        self.output_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(0.1)
        )

    def run_health_check(self) -> bool:
        all_ok = True
        for i, layer in enumerate(self.layers):
            health = layer.verify_architecture_health()
            if not all(health.values()):
                failed = [k for k, v in health.items() if not v]
                logging.error(f"[ARCH HEALTH] HybridLayer {i} failed checks: {failed}")
                all_ok = False
            else:
                logging.info(f"[ARCH HEALTH] HybridLayer {i}: all checks passed âœ“")
        return all_ok

    def forward(self, x):
        _expected_feats = self.input_embedding[0].in_features
        assert x.dim() == 3 and x.size(-1) == _expected_feats, (
            f"HybridPerceptionBackbone got shape {tuple(x.shape)}, "
            f"expected (batch, seq_len, {_expected_feats}). Upstream feature "
            f"assembly is wrong."
        )
        batch_size = x.size(0)
        x_emb = self.input_embedding(x)              
        x_raw = self.raw_proj(raw_x)                 
        mix = torch.sigmoid(self.raw_mix_gate)        
        x = (1.0 - mix) * x_emb + mix * x_raw
        
        # === ADD POSITIONAL ENCODING ===
        seq_len = x.size(1)
        x = x + self.pe_scale * self.positional_encoding[:, :seq_len, :]
        layer_stats = []
        for layer in self.layers:
            x, stats = layer(x)  # (batch, seq_len, d_model)
            layer_stats.append(stats)
        
        # Aggregate stats across layers (average)
        avg_stats = {
            'cnn': sum(s['cnn'] for s in layer_stats) / len(layer_stats),
            'lstm': sum(s['lstm'] for s in layer_stats) / len(layer_stats),
            'transformer': sum(s['transformer'] for s in layer_stats) / len(layer_stats)
        }
        x_normed = self.pre_pool_norm(x)
        query = self.pool_query.expand(batch_size, -1, -1).contiguous()  # (batch, 1, d_model)
        attn_pooled, _ = self.attention_pool(query, x_normed, x_normed)  # (batch, 1, d_model)
        attn_pooled = attn_pooled.squeeze(1)  
        mean_pooled = x_normed.mean(dim=1)  # (batch, d_model)
        alpha = torch.sigmoid(self.pool_skip_alpha)
        pooled = alpha * attn_pooled + (1.0 - alpha) * mean_pooled
        
        # === FINAL OUTPUT ===
        output = self.output_norm(pooled)
        output = self.output_proj(output)  # (batch, d_model)

        if not hasattr(self, '_nan_output_count'):
            self._nan_output_count = 0
            self._nan_output_logged = 0
        if not torch.isfinite(output).all():
            _bad = int((~torch.isfinite(output)).sum().item())
            self._nan_output_count += _bad
            if self._nan_output_count >= max(1, 2 * self._nan_output_logged):
                logging.warning(
                    f"[HybridPerceptionBackbone] non-finite pooled output "
                    f"(cumulative bad values = {self._nan_output_count}). "
                    f"Downstream will nan_to_num these to 0."
                )
                self._nan_output_logged = self._nan_output_count

        if self.training:
            if not hasattr(self, '_health_check_step'):
                self._health_check_step = 0
            self._health_check_step += 1
            if self._health_check_step % 500 == 0:
                try:
                    self.run_health_check()
                except Exception as _hc_e:
                    logging.warning(f"[ARCH HEALTH] periodic check failed: {_hc_e}")

        return output, avg_stats  # Single unified representation, not concatenation!


#part of memory systems:
class BiologicalMemoryBank:
    REGIME_MAP = {'BULL': 0, 'BEAR': 1, 'SIDEWAYS': 2, 'VOLATILE': 3}
    
    def __init__(
        self,
        soft_capacity: int = 5000,      # Target size after consolidation
        hard_capacity: int = 10000,     # Absolute max before forced prune
        decay_exponent: float = 0.25,   # Gentler than 0.3 per validation findings
        min_retention: float = 0.005,   # Lower threshold per validation findings
        consolidation_interval: int = 50,  # Trades between "sleep" cycles
        retrieval_boost: float = 1.1,   # Strength boost on retrieval
        emotional_scale: float = 0.002, # PnL -> emotional weight scaling
        max_emotional: float = 4.0,     # Cap emotional modulation
        min_emotional: float = 0.5,     # Floor emotional modulation
        recent_protection: int = 50,    # Don't prune memories younger than this
    ):
        self.memories: list = []  # List of episode dicts
        self.soft_capacity = soft_capacity
        self.hard_capacity = hard_capacity
        self.decay_exponent = decay_exponent
        self.min_retention = min_retention
        self.consolidation_interval = consolidation_interval
        self.retrieval_boost = retrieval_boost
        self.emotional_scale = emotional_scale
        self.max_emotional = max_emotional
        self.min_emotional = min_emotional
        self.recent_protection = recent_protection
        
        # Internal state
        self.current_trade_idx = 0
        self.trades_since_consolidation = 0
        self.consolidation_count = 0
        
    def __len__(self):
        return len(self.memories)
    
    def __iter__(self):
        return iter(self.memories)
    
    def __getitem__(self, idx):
        return self.memories[idx]
    
    @staticmethod
    def encode_regime_onehot(market_regime: str, device=None) -> torch.Tensor:
        """Convert regime string to 4-dim one-hot tensor."""
        onehot = torch.zeros(4, device=device)
        idx = BiologicalMemoryBank.REGIME_MAP.get(market_regime, 2)  # Default SIDEWAYS
        onehot[idx] = 1.0
        return onehot
    
    @staticmethod
    def enhance_context(context: torch.Tensor, market_regime: str) -> torch.Tensor:
        # CRITICAL FIX: Always use CPU for memory storage efficiency
        if isinstance(context, torch.Tensor):
            context_cpu = context.cpu() if context.device.type != 'cpu' else context
        else:
            context_cpu = torch.tensor(context, dtype=torch.float32)
        
        regime_onehot = BiologicalMemoryBank.encode_regime_onehot(market_regime, device='cpu')
        return torch.cat([context_cpu, regime_onehot])
    
    def calculate_emotional_weight(self, pnl: float, surprise: float = 0.0) -> float:
        # Base emotional weight from PnL magnitude
        pnl_weight = 1.0 + abs(pnl) * self.emotional_scale
        
        # Surprise bonus (prediction errors are valuable for learning)
        surprise_bonus = 1.0 + surprise * 0.5
        
        # Combined emotional weight
        emotional = pnl_weight * surprise_bonus
        
        # Clamp to valid range
        return max(self.min_emotional, min(self.max_emotional, emotional))
    
    def calculate_retention(self, episode: dict) -> float:
        age = max(1, self.current_trade_idx - episode.get('trade_idx', 0))
        
        # Power law decay (Ebbinghaus/Anderson & Schooler)
        age_decay = age ** (-self.decay_exponent)
        
        # Base strength (boosted by retrievals)
        base_strength = episode.get('strength', 1.0)
        
        # Emotional modulation
        emotional_weight = episode.get('emotional_weight', 1.0)
        
        # Spacing effect (diminishing returns on repeated access)
        access_count = episode.get('access_count', 0)
        spacing_bonus = min(1.5, 1.0 + 0.1 * min(5, access_count))
        
        # Combined retention
        retention = base_strength * emotional_weight * spacing_bonus * age_decay
        
        return retention
    
    def append(self, episode: dict, apply_interference: bool = True):
        # Update trade index
        self.current_trade_idx = episode.get('trade_idx', self.current_trade_idx + 1)
        if 'context' in episode and len(self.memories) >= 10:
            try:
                new_ctx = episode['context']
                if isinstance(new_ctx, torch.Tensor):
                    new_vec = new_ctx.detach().float().flatten()
                else:
                    new_vec = torch.tensor(new_ctx, dtype=torch.float32).flatten()
                
                new_norm = new_vec.norm()
                if new_norm > 1e-6:
                    new_vec_normed = new_vec / new_norm
                    # Check against last 10 memories
                    n_similar = 0
                    n_checked = 0
                    for recent_mem in self.memories[-10:]:
                        recent_ctx = recent_mem.get('context')
                        if recent_ctx is None:
                            continue
                        if isinstance(recent_ctx, torch.Tensor):
                            recent_vec = recent_ctx.detach().float().flatten()
                        else:
                            recent_vec = torch.tensor(recent_ctx, dtype=torch.float32).flatten()
                        
                        recent_norm = recent_vec.norm()
                        if recent_norm > 1e-6 and recent_vec.shape == new_vec.shape:
                            cos_sim = (new_vec_normed * (recent_vec / recent_norm)).sum().item()
                            n_checked += 1
                            if cos_sim > 0.98:
                                n_similar += 1
                    
                    if (n_checked >= 3 and n_similar / n_checked > 0.5 and 
                        abs(episode.get('pnl', 0)) < 200):
                        return  # Skip redundant memory
            except Exception:
                pass  # On any error, just store normally
        
        if 'context' in episode and episode.get('market_regime'):
            original_context = episode['context']
            enhanced = self.enhance_context(original_context, episode['market_regime'])
            episode['context_enhanced'] = enhanced.cpu() if isinstance(enhanced, torch.Tensor) else enhanced
            # Keep original for backward compatibility
        
        # Calculate emotional weight
        pnl = episode.get('pnl', 0.0)
        surprise = episode.get('surprise', 0.0)
        episode['emotional_weight'] = self.calculate_emotional_weight(pnl, surprise)
        
        # Initialize strength factors
        episode['strength'] = 1.0
        episode['access_count'] = 0
        episode['last_accessed'] = self.current_trade_idx
        episode['consolidated'] = False
        
        # Add to memory
        self.memories.append(episode)
        self.trades_since_consolidation += 1
        
        # Phase 2: Apply retroactive interference
        # New memory weakens similar existing memories
        if apply_interference and 'context' in episode:
            self.apply_interference(episode['context'])
        
        # Periodic consolidation (like sleep)
        if self.trades_since_consolidation >= self.consolidation_interval:
            self.consolidate()
        
        # Hard capacity enforcement (emergency prune)
        if len(self.memories) > self.hard_capacity:
            self._force_prune()
    
    def consolidate(self):
        self.trades_since_consolidation = 0
        self.consolidation_count += 1
        
        # Only prune if over soft capacity
        if len(self.memories) <= self.soft_capacity:
            return
        
        # Calculate how many to remove
        excess = len(self.memories) - self.soft_capacity
        
        # Score all memories by retention
        scored = []
        for i, mem in enumerate(self.memories):
            age = self.current_trade_idx - mem.get('trade_idx', 0)
            retention = self.calculate_retention(mem)
            scored.append({
                'idx': i,
                'retention': retention,
                'age': age,
                'protected': age < self.recent_protection
            })
        
        # Sort by retention (lowest first)
        scored.sort(key=lambda x: x['retention'])
        
        # Collect indices to remove (skip protected)
        to_remove = []
        for item in scored:
            if len(to_remove) >= excess:
                break
            if not item['protected']:
                to_remove.append(item['idx'])
        
        # Remove in reverse order to preserve indices
        for idx in sorted(to_remove, reverse=True):
            del self.memories[idx]
        
        # Mark remaining as consolidated
        for mem in self.memories:
            mem['consolidated'] = True
    
    def _force_prune(self):
        """Emergency hard capacity enforcement - removes lowest retention memories."""
        excess = len(self.memories) - self.hard_capacity
        if excess <= 0:
            return
        
        # Score and remove lowest retention (ignore protection for hard limit)
        scored = [(i, self.calculate_retention(m)) for i, m in enumerate(self.memories)]
        scored.sort(key=lambda x: x[1])
        
        to_remove = [idx for idx, _ in scored[:excess]]
        for idx in sorted(to_remove, reverse=True):
            del self.memories[idx]
    
    def on_retrieval(self, retrieved_indices: list, all_similarity_scores: torch.Tensor = None):
        retrieved_set = set(retrieved_indices)
        max_strength = 100.0
        ACCESS_CAP = 500  # After this, reduce recall priority
        
        # 1. Strengthen retrieved memories (with cap + access fatigue)
        for idx in retrieved_indices:
            if 0 <= idx < len(self.memories):
                mem = self.memories[idx]
                access = mem.get('access_count', 0)
                
                # Access fatigue: diminishing strengthening for over-accessed memories
                if access < ACCESS_CAP:
                    boost = self.retrieval_boost
                else:
                    # Logarithmic diminishing: 500 accesses = full boost, 5000 = ~0.5x
                    #boost = self.retrieval_boost * min(1.0, ACCESS_CAP / max(access, 1))
                    boost = 1.0 + (self.retrieval_boost - 1.0) * (ACCESS_CAP / max(access, 1))
                
                new_strength = mem.get('strength', 1.0) * boost
                mem['strength'] = min(new_strength, max_strength)  # CAP at max
                mem['access_count'] = access + 1
                mem['last_accessed'] = self.current_trade_idx
        
        # 2. Retrieval-induced forgetting (inhibit similar-but-not-retrieved)
        # Only apply if we have similarity scores
        if all_similarity_scores is not None and len(all_similarity_scores) == len(self.memories):
            inhibition_factor = 0.98  # Slight decay for inhibited memories
            similarity_threshold = 0.7  # Only inhibit if similarity > threshold
            
            for idx in range(len(self.memories)):
                if idx not in retrieved_set:
                    sim = all_similarity_scores[idx].item() if isinstance(all_similarity_scores[idx], torch.Tensor) else all_similarity_scores[idx]
                    if sim > similarity_threshold:
                        # Memory was similar but not retrieved - inhibit it
                        mem = self.memories[idx]
                        mem['strength'] = mem.get('strength', 1.0) * inhibition_factor
    
    def reconsolidate(self, retrieved_indices: list, current_outcome: dict, blend_rate: float = 0.1):
        if not current_outcome or 'pnl' not in current_outcome:
            return
        
        current_pnl = current_outcome.get('pnl', 0.0)
        current_direction = current_outcome.get('direction', None)
        
        for idx in retrieved_indices:
            if 0 <= idx < len(self.memories):
                mem = self.memories[idx]
                
                # Only reconsolidate if same direction (similar trade type)
                if current_direction and mem.get('direction') == current_direction:
                    old_pnl = mem.get('pnl', 0.0)
                    
                    # Blend old prediction with current reality
                    # This makes the memory more accurate for future predictions
                    new_pnl = (1 - blend_rate) * old_pnl + blend_rate * current_pnl
                    
                    # Update the memory
                    mem['pnl'] = new_pnl
                    mem['reconsolidation_count'] = mem.get('reconsolidation_count', 0) + 1
                    
                    # Update outcome label if PnL crossed zero
                    if old_pnl > 0 and new_pnl <= 0:
                        mem['outcome'] = 'loss'
                    elif old_pnl <= 0 and new_pnl > 0:
                        mem['outcome'] = 'win'
    
    def apply_interference(self, new_context: torch.Tensor, interference_factor: float = 0.1):
        if len(self.memories) < 2:
            return
        similarity_threshold = 0.8
        window = self.consolidation_interval * 2  # 100 by default
        start = max(0, len(self.memories) - window - 1)
        
        for mem in self.memories[start:-1]:  # Recent window only, exclude just-added
            ctx = mem.get('context')
            if ctx is None:
                continue
            
            # Calculate cosine similarity
            if isinstance(ctx, torch.Tensor) and isinstance(new_context, torch.Tensor):
                # CRITICAL FIX: Ensure both tensors are on the same device (CPU for memory efficiency)
                ctx_cpu = ctx.cpu() if ctx.device.type != 'cpu' else ctx
                new_cpu = new_context.cpu() if new_context.device.type != 'cpu' else new_context
                ctx_flat = ctx_cpu.flatten()
                new_flat = new_cpu.flatten()
                if ctx_flat.shape == new_flat.shape:
                    sim = torch.nn.functional.cosine_similarity(
                        ctx_flat.unsqueeze(0), 
                        new_flat.unsqueeze(0)
                    ).item()
                    
                    if sim > similarity_threshold:
                        # Apply interference decay
                        decay = 1.0 - (interference_factor * (sim - similarity_threshold) / (1.0 - similarity_threshold))
                        mem['strength'] = mem.get('strength', 1.0) * decay
    
    def get_state_dict(self) -> dict:
        """Export memory state for checkpoint saving."""
        return {
            'memories': self.memories,
            'current_trade_idx': self.current_trade_idx,
            'consolidation_count': self.consolidation_count,
            'trades_since_consolidation': self.trades_since_consolidation,
        }
    
    def load_state_dict(self, state_dict: dict):
        """Restore memory state from checkpoint."""
        self.memories = state_dict.get('memories', [])
        self.current_trade_idx = state_dict.get('current_trade_idx', 0)
        self.consolidation_count = state_dict.get('consolidation_count', 0)
        self.trades_since_consolidation = state_dict.get('trades_since_consolidation', 0)
    
    @classmethod
    def from_legacy_deque(cls, legacy_deque, current_trade_idx: int = 0) -> 'BiologicalMemoryBank':
        bank = cls()
        bank.current_trade_idx = current_trade_idx
        
        for episode in legacy_deque:
            # Preserve original episode data
            migrated = dict(episode)
            
            # Add biological memory fields if missing
            if 'emotional_weight' not in migrated:
                pnl = migrated.get('pnl', 0.0)
                surprise = migrated.get('surprise', 0.0)
                migrated['emotional_weight'] = bank.calculate_emotional_weight(pnl, surprise)
            
            if 'strength' not in migrated:
                migrated['strength'] = 1.0
            
            if 'access_count' not in migrated:
                migrated['access_count'] = 0
            
            if 'last_accessed' not in migrated:
                migrated['last_accessed'] = migrated.get('trade_idx', 0)
            
            if 'consolidated' not in migrated:
                migrated['consolidated'] = False
            
            # Add enhanced context with regime one-hot
            if 'context' in migrated and 'context_enhanced' not in migrated:
                regime = migrated.get('market_regime', 'SIDEWAYS')
                migrated['context_enhanced'] = cls.enhance_context(
                    migrated['context'], regime
                ).cpu()
            
            bank.memories.append(migrated)
        
        return bank
    
    def get_diagnostics(self) -> dict:
        """Return diagnostic information about memory state."""
        if len(self.memories) == 0:
            return {'count': 0, 'message': 'Empty memory bank'}
        
        retentions = [self.calculate_retention(m) for m in self.memories]
        ages = [self.current_trade_idx - m.get('trade_idx', 0) for m in self.memories]
        
        # Count prunable (below min_retention)
        prunable = sum(1 for r in retentions if r < self.min_retention)
        
        return {
            'count': len(self.memories),
            'current_trade_idx': self.current_trade_idx,
            'consolidation_count': self.consolidation_count,
            'retention_min': min(retentions),
            'retention_max': max(retentions),
            'retention_mean': sum(retentions) / len(retentions),
            'age_min': min(ages),
            'age_max': max(ages),
            'age_mean': sum(ages) / len(ages),
            'prunable_count': prunable,
            'prunable_pct': prunable / len(self.memories),
            'protected_count': sum(1 for a in ages if a < self.recent_protection),
        }


class EpisodicMemoryBank(nn.Module):
    def __init__(self, memory_dim=128, capacity=5000):
        super().__init__()
        self.capacity = capacity
        self.memory_dim = memory_dim
        self.query_encoder = nn.Sequential(
            nn.Linear(memory_dim, memory_dim),
            nn.LayerNorm(memory_dim),
            nn.GELU(),
        )

        self._gate_score_ema = 0.3       # Running EMA of max cosine similarity
        self._gate_score_var_ema = 0.01  # Running EMA of score variance  
        self._gate_ema_alpha = 0.05      # Smoothing factor (slow adaptation)
        self.key_encoder = nn.Sequential(
            nn.Linear(memory_dim, memory_dim),
            nn.LayerNorm(memory_dim),
            nn.GELU(),
        )
        self.value_encoder = nn.Sequential(
            nn.Linear(memory_dim, memory_dim),
            nn.LayerNorm(memory_dim),
            nn.GELU(),
        )

        self.cnn_query_proj  = nn.Sequential(nn.Linear(memory_dim, memory_dim), nn.LayerNorm(memory_dim), nn.GELU())
        self.lstm_query_proj = nn.Sequential(nn.Linear(memory_dim, memory_dim), nn.LayerNorm(memory_dim), nn.GELU())
        self.trans_query_proj = nn.Sequential(nn.Linear(memory_dim, memory_dim), nn.LayerNorm(memory_dim), nn.GELU())

        self.cnn_value_head   = nn.Sequential(nn.Linear(memory_dim, memory_dim), nn.LayerNorm(memory_dim), nn.GELU())
        self.lstm_value_head  = nn.Sequential(nn.Linear(memory_dim, memory_dim), nn.LayerNorm(memory_dim), nn.GELU())
        self.trans_value_head = nn.Sequential(nn.Linear(memory_dim, memory_dim), nn.LayerNorm(memory_dim), nn.GELU())

    def retrieve_similar(self, current_context, memory_bank, top_k=5, current_trade_idx=None, pathway_reps=None):
        _enc_dtype = next(self.query_encoder.parameters()).dtype
        current_context = current_context.to(dtype=_enc_dtype)

        if len(memory_bank) == 0:
            # Return zeros for all outputs if memory is empty
            batch_size = current_context.size(0)
            device = current_context.device
            _z = torch.zeros_like(current_context)   # (batch, dim)
            _z1 = torch.zeros((batch_size, 1), device=device)
            return (
                _z.clone(),    # retrieved_context
                _z1.clone(),   # retrieved_outcome
                _z1.clone(),   # outcome_variance
                None,          # top_k_scores
                None,          # top_k_indices
                None,          # top_k_vectors
                None,          # top_k_outcomes
                _z.clone(),    # query
                _z1.clone(),   # gating_mask
                {              # per_pathway_retrieved
                    'cnn': _z.clone(),
                    'lstm': _z.clone(),
                    'transformer': _z.clone(),
                },
            )

        # Encode query
        query = self.query_encoder(current_context)  # (batch, dim)

        if pathway_reps is not None:
            _pw_queries = [query]
            for _proj, _key in [(self.cnn_query_proj, 'cnn'),
                                 (self.lstm_query_proj, 'lstm'),
                                 (self.trans_query_proj, 'transformer')]:
                _rep = pathway_reps.get(_key)
                if _rep is not None:
                    _pw_queries.append(_proj(_rep.to(dtype=_enc_dtype)))
            query = torch.stack(_pw_queries, dim=0).mean(dim=0)  # (batch, dim)

        # Use chunked processing for memory efficiency (1000 episodes at a time)
        chunk_size = 1000
        all_keys = []
        all_raw_keys = []  # Raw normalized contexts for collapse-proof cosine similarity
        all_values = []
        all_raw_contexts = []  # Raw stored contexts for per-pathway value heads
        all_outcomes = []  # NEW: Store outcomes (PnL)
        all_time_weights = []
        all_retention_weights = []  # NEW: Biological memory retention scores

        # Convert to list for slicing support (works with both deque and BiologicalMemoryBank)
        memory_list = list(memory_bank)
        selected_indices = None  # Track original bank indices for top_k remapping

        if len(memory_list) > 500:
            # Smart selection: recency + significance + emotional impact (deduped)
            # 1. Last 200 for recency
            recent_set = set(range(len(memory_list) - 200, len(memory_list)))
            
            # 2. Top 200 by retention score (biologically significant memories)
            if hasattr(memory_bank, 'calculate_retention'):
                retention_scores = [(i, memory_bank.calculate_retention(ep)) for i, ep in enumerate(memory_list)]
            else:
                retention_scores = [(i, ep.get('emotional_weight', 1.0) * ep.get('strength', 1.0)) for i, ep in enumerate(memory_list)]
            retention_scores.sort(key=lambda x: x[1], reverse=True)
            retention_set = set(idx for idx, _ in retention_scores[:200])
            
            # 3. Top 100 by emotional weight (high-impact trades)
            emotional_scores = [(i, ep.get('emotional_weight', 1.0)) for i, ep in enumerate(memory_list)]
            emotional_scores.sort(key=lambda x: x[1], reverse=True)
            emotional_set = set(idx for idx, _ in emotional_scores[:100])
            
            # Deduplicate and select (max 500 total)
            selected_indices = sorted(recent_set | retention_set | emotional_set)
            memory_list = [memory_list[i] for i in selected_indices]

        time_decay_rate = 0.998

        for chunk_start in range(0, len(memory_list), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(memory_list))
            chunk_episodes = memory_list[chunk_start:chunk_end]

            # Ensure context is a tensor before stacking
            valid_contexts = []
            chunk_outcomes = [] # NEW: Store outcomes for this chunk
            chunk_retention = []  # NEW: Biological retention scores

            for ep in chunk_episodes:
                # Prefer original context (256-dim) for neural network layers
                # Enhanced context (260-dim) is for storage only, not NN input
                ctx = ep.get('context')
                pnl = ep.get('pnl', 0.0) # Get PnL
                
                # Get biological retention score if available
                emotional_weight = ep.get('emotional_weight', 1.0)
                strength = ep.get('strength', 1.0)
                retention_score = emotional_weight * strength
                chunk_retention.append(retention_score)
                
                norm_pnl = pnl / 1000.0 
                chunk_outcomes.append(norm_pnl)

                if isinstance(ctx, torch.Tensor):
                    valid_contexts.append(ctx.to(device=current_context.device))
                elif isinstance(ctx, (list, np.ndarray)):
                    valid_contexts.append(torch.tensor(ctx, device=current_context.device, dtype=torch.float32))
                else:
                    # Fallback for missing/invalid context
                    valid_contexts.append(torch.zeros(self.memory_dim, device=current_context.device))

            chunk_contexts = torch.stack(valid_contexts)  # (chunk_size, dim)
            
            # Convert outcomes to tensor
            chunk_outcomes_tensor = torch.tensor(chunk_outcomes, device=current_context.device, dtype=torch.float32).unsqueeze(1) # (chunk_size, 1)
            chunk_retention_tensor = torch.tensor(chunk_retention, device=current_context.device, dtype=torch.float32)

            chunk_raw_keys = F.normalize(chunk_contexts.to(_enc_dtype), p=2, dim=-1)
            chunk_keys = self.key_encoder(chunk_contexts)
            chunk_values = self.value_encoder(chunk_contexts)

            # Calculate time weights for this chunk (combined with biological retention)
            if current_trade_idx is not None:
                chunk_time_weights = torch.tensor([
                    time_decay_rate ** (current_trade_idx - ep.get('trade_idx', current_trade_idx))
                    for ep in chunk_episodes
                ], device=chunk_keys.device, dtype=torch.float32)
            else:
                # No trade index available, use uniform weights
                chunk_time_weights = torch.ones(len(chunk_episodes), device=chunk_keys.device)

            all_raw_keys.append(chunk_raw_keys)
            all_raw_contexts.append(chunk_contexts)  # collect for per-pathway value heads
            all_keys.append(chunk_keys)
            all_values.append(chunk_values)
            all_outcomes.append(chunk_outcomes_tensor)
            all_time_weights.append(chunk_time_weights)
            all_retention_weights.append(chunk_retention_tensor)

        # Concatenate all chunks
        keys = torch.cat(all_keys, dim=0)       # (total_episodes, dim) â€” learned
        raw_keys = torch.cat(all_raw_keys, dim=0)  # (total_episodes, dim) â€” L2-normalized, raw
        values = torch.cat(all_values, dim=0)   # (total_episodes, dim)
        outcomes = torch.cat(all_outcomes, dim=0) # (total_episodes, 1)
        time_weights = torch.cat(all_time_weights, dim=0)  # (total_episodes,)
        retention_weights = torch.cat(all_retention_weights, dim=0)  # (total_episodes,)

        # Compute similarity (attention scores) over ALL episodes
        similarity = torch.matmul(query, keys.T) / (self.memory_dim ** 0.5) 
        query_raw_norm = F.normalize(current_context, p=2, dim=-1)  # (batch, dim)
        cosine_sim = torch.matmul(query_raw_norm, raw_keys.T)  # (batch, total_episodes), range [-1, 1]

        # Calculate max cosine similarity for gating and novelty
        max_cosine_scores, _ = torch.max(cosine_sim, dim=-1, keepdim=True)
        max_score_val = max_cosine_scores.mean().item()
        self._gate_score_var_ema = (1 - self._gate_ema_alpha) * self._gate_score_var_ema + \
            self._gate_ema_alpha * (max_score_val - self._gate_score_ema) ** 2
        self._gate_score_ema = (1 - self._gate_ema_alpha) * self._gate_score_ema + \
            self._gate_ema_alpha * max_score_val
        adaptive_std = max(self._gate_score_var_ema ** 0.5, 0.02)
        adaptive_threshold = max(0.05, min(0.4, self._gate_score_ema - 0.5 * adaptive_std))
        gating_mask = torch.sigmoid((max_cosine_scores - adaptive_threshold) / adaptive_std)
        time_weights_expanded = time_weights.unsqueeze(0)  # (1, total_episodes)
        retention_weights_expanded = retention_weights.unsqueeze(0)  # (1, total_episodes)
        combined_weights = time_weights_expanded * retention_weights_expanded  # Element-wise multiply
        # Log-bias: log(weight) acts as additive prior in log-softmax space
        log_bias = torch.log(combined_weights.clamp(min=1e-6))
        similarity_with_bias = similarity + log_bias  # (batch, total_episodes)

        attention_weights = F.softmax(similarity_with_bias, dim=-1)

        # Weighted retrieval (aggregate ALL similar episodes, time-weighted)
        retrieved_context = torch.matmul(attention_weights, values)  # (batch, dim)
        
        # NEW: Retrieve Expected Outcome (Weighted Average of PnL)
        retrieved_outcome = torch.matmul(attention_weights, outcomes) # (batch, 1)
        
        
        
        diff_sq = (outcomes.T - retrieved_outcome) ** 2  # (B, N)
        
        # Weighted sum of squared differences
        # attention_weights is (B, N)
        outcome_variance = torch.sum(attention_weights * diff_sq, dim=-1, keepdim=True) 
        all_raw_contexts_cat = torch.cat(all_raw_contexts, dim=0).to(dtype=_enc_dtype)  # (N, dim)
        retrieved_cnn   = torch.matmul(attention_weights, self.cnn_value_head(all_raw_contexts_cat))
        retrieved_lstm  = torch.matmul(attention_weights, self.lstm_value_head(all_raw_contexts_cat))
        retrieved_trans = torch.matmul(attention_weights, self.trans_value_head(all_raw_contexts_cat))

        # Apply the gating mask to the retrieved context AND outcomes
        # If no good match was found, the retrieved context fades to zero
        retrieved_context = retrieved_context * gating_mask
        retrieved_outcome = retrieved_outcome * gating_mask
        outcome_variance = outcome_variance * gating_mask # If no match, variance is 0 (or should it be high? No, 0 means "no memory of risk")
        per_pathway_retrieved = {
            'cnn':         torch.nan_to_num(retrieved_cnn   * gating_mask, nan=0.0),
            'lstm':        torch.nan_to_num(retrieved_lstm  * gating_mask, nan=0.0),
            'transformer': torch.nan_to_num(retrieved_trans * gating_mask, nan=0.0),
        }
        top_k_scores, top_k_indices = torch.topk(cosine_sim, k=min(top_k, cosine_sim.size(-1)), dim=-1)

        # === VISUALIZATION DATA EXTRACTION ===
        
        # Flatten indices for gathering (batch * k)
        flat_indices = top_k_indices.view(-1)
        
        # Gather keys (vectors) - (batch * k, dim) -> (batch, k, dim)
        # Note: keys are (total_episodes, dim)
        top_k_vectors = keys.index_select(0, flat_indices).view(top_k_indices.size(0), top_k_indices.size(1), -1)
        
        # Gather outcomes - (batch * k, 1) -> (batch, k, 1)
        top_k_outcomes = outcomes.index_select(0, flat_indices).view(top_k_indices.size(0), top_k_indices.size(1), -1)

        if selected_indices is not None and top_k_indices is not None:
            flat = top_k_indices.reshape(-1)
            remapped = torch.tensor(
                [selected_indices[i.item()] for i in flat],
                dtype=top_k_indices.dtype, device=top_k_indices.device
            ).view_as(top_k_indices)
            top_k_indices = remapped

        return retrieved_context, retrieved_outcome, outcome_variance, top_k_scores, top_k_indices, top_k_vectors, top_k_outcomes, query, gating_mask, per_pathway_retrieved
