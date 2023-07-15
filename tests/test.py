import yaml
import argparse

def read_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read config file')
    parser.add_argument('--config-file', type=str, help='Path to the config file')
    args = parser.parse_args()

    if args.config_file:
        config = read_config(args.config_file)
        print(config)
    else:
        print("Please provide the path to the config file using the '--config-file' argument.")

    
