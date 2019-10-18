import pandas as pd


class ExcelFile():
    def __init__(self, file_path):
        self.writer = pd.ExcelWriter(file_path, engine='openpyxl')  # pylint: disable=abstract-class-instantiated

    def init_data(self, index, column_names):
        self.data = pd.DataFrame(index=pd.Series(map(str, index)), columns=column_names)

    def save_tab(self, tab_name, data=None):
        if data is None:
            aux = self.data
        else:
            aux = data
        aux.to_excel(self.writer, sheet_name=tab_name)

    def save_file(self):
        self.writer.close()

    def add_result(self, label, results):
        self.data.loc[str(label)] = pd.Series(results)
        # self.data.loc[label] = pd.Series(results)
