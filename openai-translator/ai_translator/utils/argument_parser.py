import argparse

class ArgumentParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Translate English PDF book to Chinese.')
        self.parser.add_argument('--config', type=str, default='config.yaml', help='Configuration file with model and API settings.')
        self.parser.add_argument('--model_type', type=str, choices=['GLMModel', 'OpenAIModel'], help='The type of translation model to use. Choose between "GLMModel" and "OpenAIModel".')
        self.parser.add_argument('--glm_model_url', type=str, help='The URL of the ChatGLM model URL.')
        self.parser.add_argument('--timeout', type=int, help='Timeout for the API request in seconds.')
        self.parser.add_argument('--openai_model', type=str, help='The model name of OpenAI Model. Required if model_type is "OpenAIModel".')
        self.parser.add_argument('--openai_api_key', type=str, help='The API key for OpenAIModel. Required if model_type is "OpenAIModel".')
        self.parser.add_argument('--book', type=str, help='PDF file to translate.')
        self.parser.add_argument('--file_format', type=str, help='The file format of translated book. Now supporting PDF and Markdown')
        self.parser.add_argument('--target_language', type=str, help='Which target_language you want to translate to?')

    def parse_arguments(self):
        args = self.parser.parse_args()
        # 直接檢查 model_type 是否被設置
        if not (args.model_type):  # 如果沒有提供 model_type
            print("No model_type provided. Switching to interactive mode.")
            args = self.interactive_mode()
        elif args.model_type == 'OpenAIModel' and not args.openai_model and not args.openai_api_key:
            self.parser.error("--openai_model and --openai_api_key are required when using OpenAIModel")
        return args

    def interactive_mode(self):
        args = argparse.Namespace()  # 創建一個空的 Namespace 對象
        args.config = input('Enter the path to the configuration file (default: config.yaml): ') or 'config.yaml'
        args.model_type = input('Enter the model type (GLMModel/OpenAIModel): ')
        if args.model_type == 'OpenAIModel':
            args.openai_model = input('Enter the OpenAI model name (default: gpt-3.5-turbo): ')
            args.openai_api_key = input('Enter the OpenAI API key: ')
        args.book = input('Enter the PDF file to translate: ')
        args.file_format = input('Enter the file format of the translated book (PDF/Markdown): ')
        args.target_language = input('Which target_language you want to translate to?: ')
        return args
