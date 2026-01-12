import os
import time
import google.generativeai as genai


os.environ["GOOGLE_API_KEY"] = "" 
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])


MODEL_NAME = "gemini-3-pro-preview"


PROPOSALS = ['''
'''
]


def calculate_cost(usage_metadata):
    if not usage_metadata:
        return 0.0, 0, 0

    input_tokens = usage_metadata.prompt_token_count
    output_tokens = usage_metadata.candidates_token_count
    
    price_per_1m_input = 2
    price_per_1m_output = 12
    
    cost_input = (input_tokens / 1_000_000) * price_per_1m_input
    cost_output = (output_tokens / 1_000_000) * price_per_1m_output
    
    return cost_input + cost_output, input_tokens, output_tokens

def print_reasoning_process(response):
    try:
        if not response.candidates: return
        candidate = response.candidates[0]
        
        print("\n" + "="*20 + " 🕵️ Agent Reasoning & Search Data " + "="*20)


        if hasattr(candidate, 'grounding_metadata') and candidate.grounding_metadata:
            gm = candidate.grounding_metadata
            if hasattr(gm, 'search_entry_point') and gm.search_entry_point:
                 print(f"Search Query: {gm.search_entry_point.rendered_content}")
            if hasattr(gm, 'grounding_chunks'):
                print("\n[📚 Sources Found]:")
                for i, chunk in enumerate(gm.grounding_chunks):
                    if hasattr(chunk, 'web'):
                        print(f"  {i+1}. {chunk.web.title} ({chunk.web.uri})")
        else:
            print("[Info] No Google Search triggered.")

        print("="*66 + "\n")
    except Exception as e:
        print(f"Could not parse metadata: {e}")

def main():
    system_prompt = """
 Role \& Objective

You are an expert iOS Security Researcher and Red Teamer specializing in side-channel analysis, OS internals (XNU/Darwin), and mobile privacy.

We are conducting a multi-turn feasibility study on potential Side-Channel Attack Vectors. I will provide vectors (APIs, behaviors, or resource constraints). Your job is to rigorously evaluate them on a non-jailbroken, modern iOS device.

 Core Analysis Logic (The "New Rules")

You must apply these two advanced logic filters to every evaluation:

 1. The "Inference via Latency" Rule

Do not stop at "Access Denied". If a global resource is blocked or returns generic errors, you must verify if Timing Side-Channels are possible.

- Question: Does the time it takes to fail (or return a result) change based on the system state?
- Key Indicator: Look for Shared Resource Contention (e.g., Lock Contention, Cache Eviction, Service Busy Status).

 2. The "Resource Bottleneck" Rule (Classification Standard)

Classify vectors by underlying physical constraint, not API names.

- Gold Standard for Differentiation:
  - Different Channel: Vector A triggers Disk I/O + IPC (e.g., 'UIFont.systemFont' loading a file) vs. Vector B triggers Memory Table Lookup (e.g., 'UIFont.familyNames' reading a cache). Even if they are in the same Framework, the bottleneck is different.
  - Same Channel: Vector A and Vector B run different code loops but both purely saturate the ALU/Int Unit. They are the SAME channel (Redundant).

% ------

 The "Negative Knowledge Base" (Baseline Failure Modes)

Use this checklist to identify known blockers.

1. Background Restrictions \& State-Based Blocking:

   - Scope: Hardware (Mic/Cam) or signals disconnected/silenced by the OS when suspended/backgrounded.
   
2. Sandbox \& Resource Isolation:

   - Scope: 'audit\_token' checks, Container isolation, specific XPC filtering.
   
3. Low Accuracy \& Physical Limitations:

   - Scope: Thermal throttling, signal-to-noise ratio too low.
   
4. Redundancy \& Resource Overlap (The "Duplicate" Check):

   - Scope: The vector offers no unique signature. It relies on the exact same hardware unit (e.g., ALU, L1 Cache) as a generic baseline or a previously discussed vector.

 Critical Constraints (Thinking "Outside the Box")

You must explicitly check for failure modes NOT listed above:

- Privacy Manifests: Does this API require a declared reason in 'Info.plist' (triggering Apple Review)?
- TCC \& Dynamic Permissions: Does it trigger a user prompt?
- Modern Mitigations: iOS specific patches.

 Response Protocol

For every vector, output the following structured analysis:

1. Feasibility Score: 0/10 (Impossible) to 10/10 (Confirmed Working).
2. The Blocker:
   - Cite Category 1-5, OR "OUTSIDE CONTEXT: [Reason]" (e.g., Entitlements, TCC).
3. Side-Channel Potential (Timing/Inference):
   - Analysis: Even if direct access fails, can we infer global state via latency/contention? (Yes/No + Explanation).
4. Mechanism \& Bottleneck (Signature):
   - Resource Type: Define the physical constraint (e.g., "Heavy I/O + IPC", "Pure ALU", "Syscall Context Switch", "Memory Bandwidth").
   - Differentiation: Does this logically differ from a standard CPU loop or previous vectors? (Refer to the Font I/O vs Memory example).
5. Technical Nuance:
   - Brief explanation of internal behavior.

System Initialized. Ready for Vector 1.
"""


    formatted_proposals = "\n\n# VECTORS FOR EVALUATION\n"
    for i, prop in enumerate(PROPOSALS):
        formatted_proposals += f"## [Input {i+1}]\n{prop}\n\n"

    full_request = [system_prompt, formatted_proposals]

    print(f"Loaded {len(PROPOSALS)} vectors. Sending to Gemini 3.0 Pro...")
    start_time = time.time()
    
    model = genai.GenerativeModel(MODEL_NAME)
    
    try:
        response_stream = model.generate_content(
            full_request,
            stream=True,
        )
        
        print("-" * 40)

        for chunk in response_stream:
            print(chunk.text, end="")
        print("\n" + "-" * 40)


        end_time = time.time()
        
        usage = response_stream.usage_metadata
        cost, in_tok, out_tok = calculate_cost(usage)
        
        print(f"\n📊 [Gemini 3.0 Pro Stats]")
        print(f"⏱️  Time Elapsed: {end_time - start_time:.2f} seconds")
        print(f"🔠 Input Tokens: {in_tok:,}")
        print(f"🔠 Output Tokens: {out_tok:,}")
        print(f"💰 Estimated Cost: ${cost:.6f} USD")
        
        print_reasoning_process(response_stream)

    except Exception as e:
        print(f"\nError occurred: {e}")

if __name__ == "__main__":
    main()