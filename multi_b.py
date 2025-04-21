import json
from pathlib import Path

class Inter:
    def __init__(self, json_file="id_names.json") -> None:
        self.json_file = json_file
        self.json_path = Path(json_file)
        self.name_list = []
        self.id_names_dict = {}
        self.name_v_dict = {}
        self.id_v_dict = {}
        self.ini()

    def ini(self) -> None:
        if self.json_path.exists():
            with open(self.json_path, "r", encoding="utf-8") as file:
                data = json.load(file)
            id_names_dict = {entry["id"]: entry["name"] for entry in data}
            self.id_names_dict = id_names_dict
            name_list = self.id_names_dict.values()
            self.name_list = [""] + list(name_list)
        
    def trans(self, id_v_dict: dict) -> dict:
        name_v_dict = {}
        for id, v in id_v_dict.items():
            name = self.name_list[id]
            name_v_dict[name] = v

        self.name_v_dict = name_v_dict
        return name_v_dict

    def inverse(self, name_v_dict: dict) -> dict:
        id_v_dict = {}
        for name, v in name_v_dict.items():
            id = self.name_list.index(name)
            id_v_dict[id] = v

        self.id_v_dict = id_v_dict
        return id_v_dict

class Process:
    def get_next(self, name_v_dict) -> dict:
        next_name_v_dict = {}
        for name, v in name_v_dict.items():
            if name.startswith("a_"):
                next_name_v_dict[name] = v + 10.001

        return next_name_v_dict

    def cover_get_next(self, data_dict: dict, data_dict_like_id_value=False) -> dict:
        inter = Inter()
        name_v_dict = data_dict
        if data_dict_like_id_value:
            name_v_dict = inter.trans(data_dict)
        next_name_v_dict = self.get_next(name_v_dict)
        next_data_dict = next_name_v_dict
        if data_dict_like_id_value:
            next_id_v_dict = inter.inverse(next_name_v_dict)
            next_data_dict = next_id_v_dict
        return next_data_dict

def sim(n=3):
    next_data = {
        "a_Doe": 1.1,
        "a_Smith": 2.02,
        "a_John": 3.3,
        "b_Brown": 4.04,
        "b_Davis": 5.5,
    }
    pro = Process()
    for i in range(n):
        next_data = pro.cover_get_next(next_data)
    return next_data

def multi(m=4):
    ss = 0
    for i in range(m):
        next_data = sim()
        ss += sum(next_data.values())
    return ss

if __name__ == "__main__":
    multi_sum = multi()
    print(f"Multi sum: {multi_sum}")
