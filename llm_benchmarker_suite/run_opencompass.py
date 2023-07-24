# llm_benchmarking_suite/run_opencompass.py
import subprocess

def run_opencompass(config_file, output_folder):
    command = f"python ../opencompass/run.py {config_file} -w {output_folder}"
    subprocess.run(command, shell=True)

if __name__ == "__main__":
    config_file_path = "configs/eval_demo.py"
    output_folder_path = "outputs/demo"
    run_opencompass(config_file_path, output_folder_path)
