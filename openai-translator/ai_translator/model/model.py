from book import ContentType

class Model:
    def make_text_prompt(self, text: str, target_language: str) -> str:
        return f"翻译为{target_language}：{text}"

    def make_table_prompt(self, table: str, target_language: str) -> str:
        """
        生成一個翻譯請求提示，要求將表格數據翻譯成指定的目標語言。

        :param table: 表格數據，以字符串形式提供。
        :param target_language: 目標語言。
        :return: 格式化的翻譯請求提示。
        """
        return f"Please translate to {target_language}, Contains titles and returns in tabular form, :\n{table}"

    def translate_prompt(self, content, target_language: str) -> str:
        if content.content_type == ContentType.TEXT:
            return self.make_text_prompt(content.original, target_language)
        elif content.content_type == ContentType.TABLE:
            return self.make_table_prompt(content.get_original_as_str(), target_language)

    def make_request(self, prompt):
        raise NotImplementedError("子类必须实现 make_request 方法")
