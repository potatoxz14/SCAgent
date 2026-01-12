import os
import time
import glob
import google.generativeai as genai


os.environ["GOOGLE_API_KEY"] = ""
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])


PAPERS_DIR = ""

MODEL_NAME = "gemini-3-pro-preview"
# ===========================================

def wait_for_files_active(files):
    print("Waiting for file processing...")
    for name in (file.name for file in files):
        file = genai.get_file(name)
        while file.state.name == "PROCESSING":
            print(".", end="", flush=True)
            time.sleep(2)
            file = genai.get_file(name)
        if file.state.name != "ACTIVE":
            raise Exception(f"File {file.name} failed to process")
    print("\nAll files are ready.")

def calculate_cost(usage_metadata):
    if not usage_metadata:
        return 0.0, 0, 0

    input_tokens = usage_metadata.prompt_token_count
    output_tokens = usage_metadata.candidates_token_count
    
    price_per_1m_input = 2
    price_per_1m_output = 12
    
    cost_input = (input_tokens / 1_000_000) * price_per_1m_input
    cost_output = (output_tokens / 1_000_000) * price_per_1m_output
    
    total_cost = cost_input + cost_output
    return total_cost, input_tokens, output_tokens

def print_reasoning_process(response):
    try:
        if not response.candidates:
            return

        candidate = response.candidates[0]
        
        print("\n" + "="*20 + " 🕵️ Agent Reasoning & Search Data " + "="*20)

        if hasattr(candidate, 'grounding_metadata') and candidate.grounding_metadata:
            gm = candidate.grounding_metadata
            if hasattr(gm, 'search_entry_point') and gm.search_entry_point:
                 print(f"Search Query Triggered: {gm.search_entry_point.rendered_content}")
            
            if hasattr(gm, 'grounding_chunks'):
                print("\n[📚 Sources Found]:")
                for i, chunk in enumerate(gm.grounding_chunks):
                    if hasattr(chunk, 'web'):
                        print(f"  {i+1}. {chunk.web.title}")
                        print(f"     🔗 {chunk.web.uri}")
        else:
            print("[Info] No Google Search triggered for this specific run.")


        print("\n[🧠 Model Output Composition]:")
        for part in candidate.content.parts:
            if hasattr(part, 'text'):
                print(f"  - Text block generated ({len(part.text)} chars)")

        
        print("="*66 + "\n")
            
    except Exception as e:
        print(f"Could not parse metadata: {e}")

def main():

    pdf_paths = glob.glob(os.path.join(PAPERS_DIR, "*.pdf"))
    
    if not pdf_paths:
        print(f"Error: No PDF files found in directory '{PAPERS_DIR}'")
        return

    print(f"Found {len(pdf_paths)} PDF files. Uploading to Gemini...")


    uploaded_files = []
    for path in pdf_paths:
        print(f"Uploading: {path}...")
        try:

            file = genai.upload_file(path, mime_type="application/pdf")
            uploaded_files.append(file)
        except Exception as e:
            print(f"Failed to upload {path}: {e}")


    try:
        wait_for_files_active(uploaded_files)
    except Exception as e:
        print(e)
        return

    system_prompt = """
# Role Definition
You are an elite Security Researcher specializing in **Mobile OS Side-Channel Attacks** and **iOS Internals** (specifically Apple XNU, SpringBoard, and Sandbox restrictions). You have deep expertise in App Fingerprinting, Process Isolation, Mach Kernel APIs, and hardware micro-architectural leaks.

# Context
I am providing you with the text/content of several academic papers regarding side-channel attacks.

# Your Mission
Your primary goal is **TARGETED DISCOVERY for App Fingerprinting (Top 100 Apps)**.
You must synthesize theoretical knowledge from the papers to identify specific side-channel vectors within the iOS Sandbox that can leak **which specific application a user is currently launching or using**.

*Core Hypothesis*: Different applications exhibit unique "Launch Signatures" and runtime footprints (e.g., Dynamic Library loading patterns, specific GPU shader compilations, distinct Neural Engine models, or unique UI layout rendering bursts). You need to find ways to measure these patterns from a strictly sandboxed malicious observer app.

# Step-by-Step Execution Plan

## Step 1: Deconstruct & OS-Mapping (Theoretical Foundation)
Analyze the provided papers. Perform "Mechanism-to-OS Mapping":
1.  **Core Principle Extraction**: What is the root cause of the leak? (e.g., "LLC Eviction", "Interrupt Latency", "System-wide Lock Contention").
2.  **App Lifecycle Contextualization**: How does launching a heavy app (like Instagram/TikTok) vs. a utility app (like Calculator) trigger this resource differently?
    * *Example*: If a paper discusses "File System Cache," ask: "Does the massive dylib loading sequence during App Start cause detectable IO latency?"
    * *Example*: If a paper discusses "GPU contention," ask: "Does the app's initial UI animation (SpringBoard zoom-in + App first frame) create a unique GPU burden?"

## Step 2: EXTREME ATTACK SURFACE MAPPING (The "Hunt")
Conduct a **BROAD yet FOCUSED** search for side-channels that correlate with **App Launch & Foreground Activity**.
**CRITICAL REQUIREMENT**: Use `Google Search` to verify API existence and known iOS system behaviors (e.g., `proc_info` limitations, `jetsam` thresholds).

1.  **Systematic Subsystem Scan (App-Centric Matrix)**:
    Focus on subsystems that fluctuate heavily when a new process is spawned and moves to the foreground. Search for vectors in EACH category:

    * **Category A: The Launch Pipeline (Execution Fingerprints)**
        * **CPU & Scheduler**: Search for "XNU scheduler side channel", "detecting high-performance core burst during app launch". *Different apps have different 'Time-to-Interactive' profiles.*
        * **Loader & Memory**: Search for "dyld shared cache side channel", "page fault side channel iOS". *Can we detect which specific system frameworks the target app is linking against?*

    * **Category B: Visual & Graphics (Rendering Fingerprints)**
        * **Compositor (RenderServer)**: Search for "BackBoardd load monitoring", "frame rate drop detection from background". *Does launching a game (Unity/Unreal) cause a distinct dip in the compositor compared to a UIKit app?*
        * **GPU & Metal**: Search for "system-wide GPU utilization inference", "Metal shader compilation side channel".

    * **Category C: Compute & Intelligence (Feature Fingerprints)**
        * **Neural Engine (ANE)**: Search for "CoreML model loading latency", "ANE exclusive access side channel". *Does the app immediately load a FaceID or Image Recognition model?*
        * **Audio/Haptics**: Search for "CoreHaptics vibration detection", "AudioSession route change detection". *Does the app play a distinct startup sound or haptic tick?*

    * **Category D: System Resources (State Fingerprints)**
        * **IPC & Services**: Search for "XPC service blocking", "mach port contention". *Does the app rely heavily on specific system daemons (e.g., `locationd`, `biometrickitd`)?*

2.  **Feasibility Check**:
    * For every vector, ask: "Is this visible to a background sandboxed app while another app is launching?" (Consider iOS background execution limits).

## Step 3: Targeted Vector Selection & Analysis Select and detail exactly 10 distinct side-channel vectors. Critically, these vectors must be NOVEL and NOT mentioned in the provided papers. You should propose new attack surfaces for website classification based on your knowledge of iOS internals.

Selection Criteria: Prioritize NOVEL and EXPERIMENTAL vectors (e.g., Apple Neural Engine (ANE) contention, GPU shader jitter, cache occlusion) over generic metrics like simple CPU usage. Ensure the selection covers diverse architectural dimensions (Computation, Rendering, Memory).

Mechanism Explanation: For each vector, explicitly explain the causal link—precisely why this vector leaks specific web content information (e.g., "Vector X correlates linearly with the complexity of CSS reflow operations").
"""


    request_content = [system_prompt]
    print("\nAnalyzing papers and generating attack vectors... (This may take a minute)")
    start_time = time.time()

    request_content.extend(uploaded_files)
    
    model = genai.GenerativeModel(MODEL_NAME)
    
    try:
        response = model.generate_content(
            request_content,
            stream=True,
            tools='google_search_retrieval' 
        )
    except Exception:
        print("Google Search tool not available or failed, running without it...")
        response = model.generate_content(request_content, stream=True)

    print("-" * 40)
    for chunk in response:
        print(chunk.text, end="")
    print("\n" + "-" * 40)

    end_time = time.time()
    try:
        usage = response.usage_metadata
        cost, in_tok, out_tok = calculate_cost(usage)
        
        print(f"\n📊 [Gemini 3.0 Pro Stats]")
        print(f"⏱️  Time Elapsed: {end_time - start_time:.2f} seconds")
        print(f"🔠 Input Tokens: {in_tok:,} (PDFs + Prompt)")
        print(f"🔠 Output Tokens: {out_tok:,} (Thinking + Response)")
        print(f"💰 Estimated Cost: ${cost:.6f} USD")
        
        print_reasoning_process(response)

    except Exception as e:
        print(f"\nError stats: {e}")

if __name__ == "__main__":
    main()