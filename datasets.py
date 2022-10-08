import joblib
import logging
import pandas
from abc import ABC, abstractmethod
from pathlib import Path

class Datasets:
    """データセットを読み込むクラス

    実質的な AbstractLoader のラッパ
    思い付きで構成したので，使いづらい

    Attributes:
        data_type: データセットの形式
        raw_data_file: 読み込むCSVファイルのパス
        nrows: 読み込む行数
        data_range: nrows以下で読み込む範囲

        load: CSV を読み込み DataFrame を返すメソッド
        sample: CSV を読み込み一部を抽出して保存
    """

    def __init__(self, raw_data_file, data_type, nrows=None, data_range=None):
        """初期化

        Args:
            data_type: データセットの形式
            raw_data_file: 読み込むCSVファイルのパス
            nrows: 読み込む行数
            data_range: nrows以下で読み込む範囲

        Returns:
            None
        """

        self.logger = logging.getLogger(__name__)

        self.data_type = data_type
        self.raw_data_file = raw_data_file
        self.nrows = nrows
        self.data_range = data_range

        self.Loader = {
                "CIDDS-001": CIDDS_001_Loader,
                "CIDDS-002": CIDDS_002_Loader,
                }[self.data_type]
        self.loader = self.Loader(
                        self.raw_data_file,
                        self.nrows,
                        self.data_range)
        self.load = self.loader.load
        self.sample = self.loader.sample

    @classmethod
    def get_unit(cls, attribute):
        unit_dict = {
        "Duration":                     "s",
        "Transport protocol":           None,
        "IP address":                   None,
        "Source IP address":            None,
        "Destination IP address":       None,
        "Port":                         None,
        "Source port":                  None,
        "Destination port":             None,
        "Number of transmitted bytes":  None,
        "Number of transmitted packets":None
        # "Number of transmitted bytes":  "byte",
        # "Number of transmitted packets":"packet"
        }
        return unit_dict[attribute]

    @classmethod
    def decode_metric_prefix(cls, x):
        if x.replace(' ', '').isnumeric():
            return int(x)
        prefixes = {"k": 1e3, "M": 1e6, "G": 1e9}
        return int(prefixes[x[-1]] * float(x[:-1]))

class AbstractLoader(ABC):
    """データセットを読み込む抽象クラス

    CSVファイルから DataFrame でデータを読み込み， 属性名を変更して返す機能を提供
    試しに Template Method パターンで実装してみた
    思い付きで構成したので，使いづらい

    Attributes:
        raw_data_file: データセットのパス
        nrows: 読み込む行数
        data_range: 範囲
        flow_dataset: DataFrame 型のフローデータ

    """

    def __init__(self, raw_data_file, nrows=None, data_range=None):
        """初期化
        """

        self.logger = logging.getLogger(__name__)
        self.raw_data_file = raw_data_file
        self.nrows = nrows
        self.data_range = data_range

    def load(self):
        """データセットの読み込み

        Returns:
            DataFrame
        """

        self.logger.info(f"Load {self.raw_data_file}")
        self.csv2df()
        self.rename_fields()
        self.format_data()

        self.logger.info(f'''Columns:
        {self.flow_dataset.columns.tolist()}''')
        return self.flow_dataset

    def csv2df(self):
        """CSV を読み込み DataFrame に変換
        """
        self.flow_dataset = pandas.read_csv(
                self.raw_data_file,
                sep=',',
                header=0,
                nrows=self.nrows,
                usecols=self.columns.keys(),
                dtype='object')

        if self.data_range is not None:
            self.flow_dataset = self.flow_dataset[self.data_range[0]:self.data_range[1]]
            self.flow_dataset.reset_index(drop=True, inplace=True)

    def rename_fields(self):
        """データセットの属性名を一般的な名前にリネーム
        """
        self.flow_dataset = self.flow_dataset.rename(
                columns=self.columns)

    def format_data(self):
        pass

    def sample(self, output_file):
        """データセットから一部を抽出して保存
        """
        self.csv2df()
        self.sample_df()
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        self.flow_dataset.to_csv(output_file)
        self.logger.info(f"Save {output_file}")

    def sample_df(self):
        pass

class CIDDS_001_Loader(AbstractLoader):
    """CIDDS-001 を読み込むクラス
    """
    columns = {
            "Date first seen":  "Date first seen",
            "Duration":         "Duration",
            "Proto":            "Transport protocol",
            "Src IP Addr":      "Source IP address",
            "Src Pt":           "Source port",
            "Dst IP Addr":      "Destination IP address",
            "Dst Pt":           "Destination port",
            "Bytes":            "Number of transmitted bytes",
            "Packets":          "Number of transmitted packets",
            "Flags":            "TCP flags",
            }

    def format_data(self):
        """CIDDS-001 の整形

        ICMPを除外
        """
        self.flow_dataset["Number of transmitted bytes"] \
                = self.flow_dataset["Number of transmitted bytes"].map(
                       Datasets.decode_metric_prefix)
        self.flow_dataset = self.flow_dataset[self.flow_dataset
                ["Transport protocol"] != "ICMP "]
        self.flow_dataset = self.flow_dataset.astype({
            "Source port": 'int32',
            "Destination port": 'int32',
            "Number of transmitted bytes": 'int32',
            "Number of transmitted packets": 'int32',
            })
        for attr in [
                "Duration",
                "Source IP address",
                "Destination IP address",
                "Transport protocol",]:
            self.flow_dataset[attr] = \
            self.flow_dataset[attr].str.strip()

        # self.flow_dataset = self.flow_dataset[self.fields]

    def sample_df(self):
        sample_attr = [f'192.168.220.{n}' for n in range(4, 16+1)] \
        + [f'192.168.200.{n}' for n in [4, 5, 8, 9]] \
        + ['192.168.100.6']
        self.flow_dataset = self.flow_dataset[
                self.flow_dataset['Src IP Addr'].isin(sample_attr)]
        self.flow_dataset = self.flow_dataset[
                self.flow_dataset['Dst IP Addr'].isin(sample_attr)]

class CIDDS_002_Loader(AbstractLoader):
    """CIDDS-002 を読み込むクラス
    """
    columns = {
            "Date first seen":  "Date first seen",
            "Duration":         "Duration",
            "Proto":            "Transport protocol",
            "Src IP Addr":      "Source IP address",
            "Src Pt":           "Source port",
            "Dst IP Addr":      "Destination IP address",
            "Dst Pt":           "Destination port",
            "Bytes":            "Number of transmitted bytes",
            "Packets":          "Number of transmitted packets",
            "Flags":            "TCP flags",
            }

    def format_data(self):
        """CIDDS-002 の整形

        ICMPを除外
        """
        self.flow_dataset["Number of transmitted bytes"] \
                = self.flow_dataset["Number of transmitted bytes"].map(
                       Datasets.decode_metric_prefix)
        self.flow_dataset = self.flow_dataset[self.flow_dataset
                ["Transport protocol"] != "ICMP "]
        self.flow_dataset = self.flow_dataset.astype({
            "Source port": 'int32',
            "Destination port": 'int32',
            "Number of transmitted bytes": 'int32',
            "Number of transmitted packets": 'int32',
            })
        for attr in [
                "Duration",
                "Source IP address",
                "Destination IP address",
                "Transport protocol",]:
            self.flow_dataset[attr] = \
            self.flow_dataset[attr].str.strip()

        # self.flow_dataset = self.flow_dataset[self.fields]

