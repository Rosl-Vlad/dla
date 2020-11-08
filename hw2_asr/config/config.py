import json


def set_config():
    with open("config/params.json", "r") as rdr:
        config = json.load(rdr)

    return config
