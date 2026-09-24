# -*- coding: utf-8 -*-
import os
import sys
import datetime
import threading
import time
import subprocess
import urllib.request
import collections

# =========================================================================
# 0. Persistent Storage Setup (/data bucket)
# =========================================================================
IS_SPACES = os.environ.get("SPACE_ID") is not None

if IS_SPACES:
    DATA_DIR = "/data/workspace_data"
    HIDDEN_DIR = "/data/.hidden_data"
    PUBLIC_DIR = "/data/public_data"
    os.environ["HF_HOME"] = "/data/models_cache"
else:
    DATA_DIR = "./local_workspace_data"
    HIDDEN_DIR = "./.hidden_data"
    PUBLIC_DIR = "./public_data"

IMAGE_DIR = os.path.join(HIDDEN_DIR, "saved_images")
FILES_DIR = os.path.join(HIDDEN_DIR, "downloads")
CHAT_LOG_FILE = os.path.join(HIDDEN_DIR, "chat_history.txt")
SETUPBOT_LOG_FILE = "/tmp/setupbot_live.log"

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(FILES_DIR, exist_ok=True)
os.makedirs(PUBLIC_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

import gradio as gr
import spaces
from smolagents import CodeAgent, InferenceClientModel, tool
from huggingface_hub import InferenceClient

# =========================================================================
# 1. AI Tools & Agent Setup (smolagents)
# =========================================================================
@tool
def execute_shell_command(command: str) -> str:
    """Runs a shell command and returns the output. Use this for git clone, pip install, ls, etc.
    
    Args:
        command: The bash command to run.
    """
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=120)
        return result.stdout if result.returncode == 0 else f"Error: {result.stderr}"
    except Exception as e:
        return f"Exception: {str(e)}"

@tool
def write_to_file(filepath: str, content: str) -> str:
    """Writes content to a file. Use this instead of python open() function.
    
    Args:
        filepath: The absolute path to save the file (e.g. /data/workspace_data/downloads/script.py)
        content: The string content to write.
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return f"File saved successfully at {filepath}"
    except Exception as e:
        return f"Failed to write file: {e}"

@tool
def read_file(filepath: str) -> str:
    """Reads content from a file.
    
    Args:
        filepath: The path to the file.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}"

# Default token from environment secrets if available
default_hf_token = os.environ.get("HF_TOKEN", None)
agent_model = InferenceClientModel("Qwen/Qwen2.5-Coder-32B-Instruct", token=default_hf_token)

agent = CodeAgent(
    tools=[execute_shell_command, write_to_file, read_file], 
    model=agent_model, 
    add_base_tools=False,
    additional_authorized_imports=["os", "requests", "json", "time", "datetime", "urllib", "zipfile"]
)

def save_chat_log(user_msg, ai_response):
    try:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(CHAT_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] USER: {user_msg}\n")
            f.write(f"[{timestamp}] AI: {ai_response}\n")
            f.write("-" * 50 + "\n")
    except Exception:
        pass

def chat_with_true_agent(message, history, secret_key="", hf_token="", build_repo_id=""):
    real_secret = secret_key or os.environ.get("MY_SECRET", "")
    if real_secret != "Akshay123":
        yield "Work in progress... Please enter valid User ID in Settings."
        return
        
    final_hf_token = hf_token or os.environ.get("HF_TOKEN", "")
    final_repo_id = build_repo_id or os.environ.get("BUILD_REPO_ID", "")
    
    if not final_hf_token:
        yield (
            "⚠️ **Hugging Face Access Token सापडला नाही!**\n\n"
            "AI Agent ला मॉडेल कॉल करण्यासाठी Hugging Face Access Token आवश्यक आहे.\n\n"
            "**कसे सोडवायचे:**\n"
            "1. Hugging Face वरील Space **Settings ⚙️** -> **Variables and secrets** मध्ये जा.\n"
            "2. **New Secret** वर क्लिक करा, Name: `HF_TOKEN` आणि Value: तुमचा HF Access Token (Read) टाका.\n"
            "3. किंवा वर **⚙️ Settings** उघडून **Access Token** बॉक्समध्ये तुमचा टोकन टाका."
        )
        return

    try:
        agent.model = InferenceClientModel("Qwen/Qwen2.5-Coder-32B-Instruct", token=final_hf_token)
    except Exception as e:
        yield f"❌ Model Setup Error: {str(e)}"
        return
    
    system_instruction = f"""
    You are an autonomous AI Agent running in a Hugging Face Python Space.
    Your working directory for saving files is: {FILES_DIR}
    
    CRITICAL INSTRUCTIONS:
    1. NEVER use the built-in python `open()` function. ALWAYS use your `write_to_file` and `read_file` tools to handle files.
    2. NEVER use `subprocess.run` or `os.system` in your python code. ALWAYS use your `execute_shell_command` tool to run bash commands like `git clone` or `pip install`.
    3. RELENTLESS PROBLEM SOLVING: If a tool, command, or approach fails, DO NOT GIVE UP. Keep trying different methods until you succeed!
    4. SELF-TESTING & ERROR FIXING: NEVER give broken code to the user! Before finishing, you MUST test your own code using your shell tool. If you see an error, FIX IT YOURSELF.
    5. INTERACTIVE PROBLEM SOLVING: If you are completely stuck (e.g., missing a token, permission denied, or need user input), DO NOT FAKE SUCCESS. Simply stop and ASK the user for the missing information. Wait for their reply, then resume.
    6. STANDALONE DOCKER BUILDER (FULL CONTROL): The user has given you a dedicated Docker Repo to use as your personal build system!
       - HF Token: {final_hf_token if final_hf_token else 'NOT PROVIDED'}
       - Build Repo ID: {final_repo_id if final_repo_id else 'NOT PROVIDED'}
       If provided, this remote repo is 100% YOURS. To push files to it, DO NOT use `git` or `hf cli` in shell (they cause auth errors). ALWAYS write a python script using `huggingface_hub.HfApi().upload_folder(repo_id=..., folder_path=..., repo_type=\"space\", token=...)` to push your files. 
       Hugging Face will automatically build it. Once your build is successful, give the user the URL to that repo to download their `.apk`.
       If no credentials are provided and they want an APK, build a Progressive Web App (PWA) with a `manifest.json`.
    7. ZIP MULTIPLE FILES: If your solution involves creating multiple files locally, zip them into a `.zip` in {FILES_DIR}.
    8. VERIFICATION: Always verify that your work was successful (e.g., read the HfApi response) before giving the final answer.
    
    User Request: {message}
    """
    
    response_text = "🧠 **Agent is starting its work...**\n"
    yield response_text
    
    try:
        steps = agent.run(system_instruction, stream=True)
        for step in steps:
            if isinstance(step, str):
                response_text += f"\n\n🎯 **Final Answer:**\n{step}"
            else:
                response_text += f"\n\n⚙️ **Step Progress:**\n```text\n{str(step)}\n```"
            yield response_text
            
        save_chat_log(message, response_text)
    except Exception as e:
        err_msg = str(e)
        if "401" in err_msg or "Unauthorized" in err_msg:
            yield (
                response_text + f"\n\n❌ **401 Unauthorized Error:**\n"
                f"तुमचा Hugging Face Token अवैध किंवा एक्सपायर झाला आहे.\n"
                f"कृपया Hugging Face Settings -> Access Tokens मधून नवीन टोकन बनवा आणि "
                f"Space च्या **Settings -> Variables and secrets -> HF_TOKEN** मध्ये सेव्ह करा.\n"
                f"त्रुटी तपशील: `{err_msg}`"
            )
        else:
            yield response_text + f"\n\n❌ Agent ला काम करताना अडचण आली: {err_msg}"

# =========================================================================
# 2. AI Media Creation Setup (Ultra-Fast FLUX API)
# =========================================================================
IMAGE_MODEL_ID = "black-forest-labs/FLUX.1-schnell"

def get_saved_images():
    return []

def generate_media(prompt):
    try:
        curr_token = os.environ.get("HF_TOKEN", None)
        client = InferenceClient(token=curr_token)
        image = client.text_to_image(prompt, model=IMAGE_MODEL_ID)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(IMAGE_DIR, f"gen_{timestamp}.png")
        image.save(save_path)
        
        return image, f"✅ Success! Image saved to: {save_path}", get_saved_images()
    except Exception as e:
        return None, f"❌ Error: {str(e)}", get_saved_images()

# =========================================================================
# 3. File Explorer (Download/Upload)
# =========================================================================
def list_files():
    all_files = []
    for root, dirs, files in os.walk(PUBLIC_DIR):
        for file in files:
            all_files.append(os.path.join(root, file))
    return all_files

def save_text_to_file(filename, content):
    if not filename: return "Please enter a filename", list_files()
    path = os.path.join(FILES_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"✅ Saved {filename} to {path}", list_files()

@spaces.GPU
def wake_up_gpu():
    return "✅ Local ZeroGPU is successfully allocated!"

# =========================================================================
# 4. Background Setupbot & Git Fetch Runner with Live Logs
# =========================================================================
SETUPBOT_REPO_URL = "https://github.com/AnataVortex7/setup.git"
SETUPBOT_PID_FILE = "/root/setupbot.pid"
setupbot_log_buffer = collections.deque(maxlen=500)

def append_setupbot_log(text):
    timestamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    line = f"{timestamp} {text}\n"
    setupbot_log_buffer.append(line)
    try:
        with open(SETUPBOT_LOG_FILE, "a", encoding="utf-8") as lf:
            lf.write(line)
    except Exception:
        pass

def get_live_setupbot_logs():
    try:
        if os.path.exists(SETUPBOT_LOG_FILE):
            with open(SETUPBOT_LOG_FILE, "r", encoding="utf-8") as lf:
                lines = lf.readlines()
                return "".join(lines[-300:])
    except Exception as e:
        return f"Error reading logs: {e}"
    return "".join(setupbot_log_buffer) or "⏳ No logs yet. Bot is starting..."

def launch_setupbot_background():
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not bot_token:
        append_setupbot_log("❌ [Setupbot] TELEGRAM_BOT_TOKEN is not set in Secrets. Bot will stay idle.")
        return

    append_setupbot_log("🚀 [Setupbot] Cloning/Fetching setupbot from GitHub repository...")
    target_dir = "/tmp/gh_setup_repo"
    target_script = "/root/setupbot.py"

    try:
        if os.path.exists(target_dir):
            subprocess.run(f"cd {target_dir} && git pull origin main", shell=True, check=True)
        else:
            subprocess.run(f"git clone {SETUPBOT_REPO_URL} {target_dir}", shell=True, check=True)
        
        src_file = os.path.join(target_dir, "setupbot.py")
        if os.path.exists(src_file):
            with open(src_file, "r", encoding="utf-8") as sf:
                code = sf.read()
            
            # Patch base64 bot code to use root directory /
            import re
            import base64
            match = re.search(r'BOT_CODE_B64 = b"([^"]+)"', code)
            if match:
                b64_str = match.group(1)
                decoded_bot = base64.b64decode(b64_str).decode('utf-8')
                decoded_bot = decoded_bot.replace('WORKSPACE = "/data/workspace"', 'WORKSPACE = "/"')
                decoded_bot = decoded_bot.replace('WORKSPACE = os.path.expanduser("~/workspace")', 'WORKSPACE = "/"')
                decoded_bot = decoded_bot.replace('os.path.expanduser("~/workspace/backup_part_*")', '"/backup_part_*"')
                decoded_bot = decoded_bot.replace('~/workspace/backup_part_*', '/backup_part_*')
                
                new_b64 = base64.b64encode(decoded_bot.encode('utf-8')).decode('utf-8')
                code = code.replace(b64_str, new_b64)
            
            with open(target_script, "w", encoding="utf-8") as tf:
                tf.write(code)
            append_setupbot_log(f"✅ [Setupbot] Successfully fetched and patched setupbot.py to {target_script}")
        else:
            append_setupbot_log("⚠️ [Setupbot] setupbot.py not found in cloned repo root, checking fallback...")
    except Exception as e:
        append_setupbot_log(f"❌ [Setupbot] Git fetch error: {e}")

    if not os.path.exists(target_script) and os.path.exists("./setupbot.py"):
        target_script = "./setupbot.py"

    if os.path.exists(target_script):
        env = os.environ.copy()
        env["TELEGRAM_BOT_TOKEN"] = bot_token
        env["TELEGRAM_ALLOWED_USER_ID"] = os.environ.get("TELEGRAM_ADMIN_ID", "1193564058")
        env["IS_DOCKER"] = "1"
        try:
            append_setupbot_log(f"⚙️ [Setupbot] Spawning process for {target_script}...")
            proc = subprocess.Popen(
                [sys.executable, target_script, "--foreground"],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            try:
                with open(SETUPBOT_PID_FILE, "w") as pf:
                    pf.write(str(proc.pid))
            except Exception as e:
                append_setupbot_log(f"[Setupbot] Could not write PID file: {e}")
            
            append_setupbot_log(f"✅ [Setupbot] Bot process started successfully! PID={proc.pid}")

            # Stream process output to log file
            for line in proc.stdout:
                append_setupbot_log(f"[BotProc] {line.strip()}")
                
        except Exception as e:
            append_setupbot_log(f"❌ [Setupbot] Launch error: {e}")
    else:
        append_setupbot_log(f"❌ [Setupbot] Could not find {target_script} to launch.")

def force_update_and_restart_bot():
    try:
        if os.path.exists(SETUPBOT_PID_FILE):
            try:
                old_pid = int(open(SETUPBOT_PID_FILE).read().strip())
                os.kill(old_pid, 9)
                append_setupbot_log(f"🛑 Killed old bot PID {old_pid}")
            except Exception:
                pass
            try:
                os.remove(SETUPBOT_PID_FILE)
            except Exception:
                pass

        os.system("pkill -9 -f telegram_bot.py 2>/dev/null || true")
        os.system("pkill -9 -f setupbot.py 2>/dev/null || true")
        time.sleep(1)

        threading.Thread(target=launch_setupbot_background, daemon=True).start()
        return "✅ Bot restarted successfully via GitHub fetch!"
    except Exception as e:
        return f"❌ Error restarting bot: {e}"

# =========================================================================
# 5. Gradio UI Interface
# =========================================================================
custom_css = "#component-0 { max-width: 1100px; margin: auto; }"

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.Markdown("<h1 style='text-align: center;'>🚀 AI Assistant & Autonomous SetupBot VM</h1>")
    
    with gr.Tabs():
        # Tab 1: AI App Creator
        with gr.TabItem("👨‍💻 AI App Creator & Agent"):
            with gr.Accordion("⚙️ Settings & Bot Control", open=True):
                with gr.Row():
                    secret_box = gr.Textbox(label="User ID", type="password", placeholder="Enter ID...")
                    hf_token_box = gr.Textbox(label="Access Token", type="password", placeholder="Optional HF token...")
                    repo_id_box = gr.Textbox(label="Workspace ID", placeholder="Optional workspace repo...")
                with gr.Row():
                    update_bot_btn = gr.Button("🔄 Git Fetch & Restart SetupBot", variant="primary")
                    update_bot_status = gr.Textbox(label="Status", interactive=False)
            
            chat_interface = gr.ChatInterface(
                fn=chat_with_true_agent, 
                chatbot=gr.Chatbot(height=450),
                additional_inputs=[secret_box, hf_token_box, repo_id_box]
            )
            update_bot_btn.click(fn=force_update_and_restart_bot, inputs=[], outputs=[update_bot_status])
            
        # Tab 2: Live SetupBot Logs
        with gr.TabItem("📜 SetupBot Live Logs & Errors"):
            gr.Markdown("### 📡 setupbot.py Real-Time Execution Logs & Errors")
            log_output = gr.Textbox(label="Live Logs", value=get_live_setupbot_logs(), lines=25, max_lines=40, interactive=False)
            refresh_log_btn = gr.Button("🔄 Refresh Logs", variant="primary")
            refresh_log_btn.click(fn=get_live_setupbot_logs, inputs=[], outputs=[log_output])
            
        # Tab 3: Media Creation
        with gr.TabItem("🎨 AI Media Creator"):
            gr.Markdown("### 🖼️ टेक्स्ट मधून फोटो बनवा (FLUX.1-schnell)")
            with gr.Row():
                with gr.Column(scale=1):
                    media_prompt = gr.Textbox(label="Prompt", placeholder="A futuristic cyber city...", lines=3)
                    gen_btn = gr.Button("Generate & Save 💾", variant="primary")
                    gpu_btn = gr.Button("🔋 Wake Up ZeroGPU (Optional)", size="sm")
                    status_out = gr.Textbox(label="Status", interactive=False)
                with gr.Column(scale=1):
                    media_out = gr.Image(label="Current Generation")
                    
            gr.Markdown("### 📂 Your Saved Gallery")
            gallery = gr.Gallery(label="Saved Images", value=get_saved_images(), columns=4, height=300)
            refresh_img_btn = gr.Button("🔄 Refresh Gallery")
            
            gen_btn.click(fn=generate_media, inputs=[media_prompt], outputs=[media_out, status_out, gallery])
            gpu_btn.click(fn=wake_up_gpu, inputs=[], outputs=[status_out])
            refresh_img_btn.click(fn=get_saved_images, inputs=[], outputs=[gallery])

        # Tab 4: File Explorer
        with gr.TabItem("📁 File Explorer & Downloads"):
            gr.Markdown("### 📥 Persistent Storage मधील फाईल्स डाउनलोड करा")
            file_list = gr.File(label="Your Files in /data", value=list_files(), file_count="multiple", interactive=False)
            refresh_file_btn = gr.Button("🔄 Refresh File List")
            
            gr.Markdown("### ✍️ Save Code to File")
            with gr.Row():
                file_name = gr.Textbox(label="File Name (उदा. app.py, script.sh)")
                file_content = gr.Code(label="Code / Content", language="python")
            save_file_btn = gr.Button("Save to Downloads 💾")
            save_status = gr.Textbox(label="Status")
            
            refresh_file_btn.click(fn=list_files, inputs=[], outputs=[file_list])
            save_file_btn.click(fn=save_text_to_file, inputs=[file_name, file_content], outputs=[save_status, file_list])

if __name__ == "__main__":
    threading.Thread(target=launch_setupbot_background, daemon=True).start()
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft(), css=custom_css, ssr_mode=False)
