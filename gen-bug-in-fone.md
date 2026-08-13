# Diagnosis of FoNE-DP and FoNE Failures

I have thoroughly inspected the results in `fone_dp_totto-200.json` and the `gpt2_fone_dp.py` code. I found **two massive architectural and code issues** that perfectly explain why the model outputs are still bad, why hallucination loops happen, and why the Numeric Exact Match plummeted to 40%.

## 1. The Inference Bug (The Cause of Hallucination Loops)

Both our original FoNE model and the new FoNE-DP model suffer from a critical bug in the `generate()` function. 

During training, every `[NUM]` token is accompanied by its FoNE embedding via the `number_values` array. The model learns to expect this mathematical embedding alongside the `[NUM]` token.

However, look at what happens during generation in `gpt2_fone_dp.py` (and `gpt2_fone.py`):
```python
# We predict the token and decode the value
if next_token == [NUM]:
    pred_val = self.fone.decode(proj_out)

# BUT we append 0.0 to the sequence for the next step!
input_ids = torch.cat([input_ids, next_token], dim=1)
new_nv    = torch.zeros(B, 1)          # <-- BUG! This should be pred_val
new_nm    = torch.zeros(B, 1, dtype=torch.bool) # <-- BUG! This should be True
```
**Impact:** Whenever the model generates a number, we append the `[NUM]` token to the context, but we feed it a value of `0.0` and turn the `number_mask` to `False`. The model sees a "bare" `[NUM]` token with no FoNE embedding. This is completely out-of-distribution! It confuses the language model so badly that it panics and just guesses `[NUM]` again, creating the infinite hallucination loop.

This bug alone destroys the BLEU score because all text *after* the first generated number becomes garbage.

## 2. The Detach Trade-off (Why Exact Match dropped to 40%)

The original FoNE model had a 72% Numeric Exact Match, but FoNE-DP dropped to 40%. Why?

In FoNE-DP, we added `.detach()` before the projection head:
```python
h_detached = num_hidden.detach()
proj_out = self.num_proj(h_detached)
num_loss = self.fone.compute_loss(proj_out, tgt_vals)
```
While this successfully shielded the backbone from the "Representation Conflict," it also **removed the incentive for the backbone to remember the number**. 
Because the gradient doesn't flow back, the transformer backbone says: *"My only job is to predict the `[NUM]` token. I don't need to pass the exact numerical value to the hidden state."*

As a result, `num_proj` is forced to guess the number blindly based only on the linguistic context. For example, in the results file:
*   **Reference:** "... 76th Governor of South Carolina from 1874."
*   **FoNE-DP:** "... 67th Governor of South Carolina in 1879..."

It guessed 67 and 1879 because they are plausible contextually, but it lost the exact values because the backbone didn't pass them forward!

---

## Proposed Fixes and Next Steps

We need to fix both issues to see the true potential of these models.

1. **Fix the Inference Bug:** I will patch `generate()` in both `gpt2_fone.py` and `gpt2_fone_dp.py` to correctly inject `pred_val` into `new_nv` and set `new_nm = True`.
2. **Remove `.detach()` from FoNE-DP:** To get the Numeric Exact Match back up, we must allow the gradient to flow. However, to soften the Representation Conflict, we will keep the `num_proj` MLP! This transforms FoNE-DP into a standard **Multi-Task Projection Head**. The 99k parameter MLP will absorb the complex geometric transformation of the digit-wise loss, taking the pressure off the backbone's hidden state, without blinding it entirely.

Once I apply these two fixes, we should re-evaluate. The hallucination loops should disappear, and the exact match should jump back up. 

Shall I proceed with these code fixes?
