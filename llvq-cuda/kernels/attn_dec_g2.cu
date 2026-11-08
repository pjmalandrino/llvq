// attn_dec_g2.cu: attn_dec.cu's body at 2 query heads per KV head, Qwen3-0.6B's
// shape (16 query heads, 8 KV heads), the speculative decode's drafter. Appended
// to the module only for a model of that shape (attn.rs::AttnModule::on_stream_for),
// so the served module's text and digest do not move. The card checks it
// against the CPU replay bit for bit at every load, as it does the others.
ATTN_ENTRY(attn_dec_g2_u64, 2, 64)
ATTN_ENTRY(attn_dec_g2_u32, 2, 32)
