"""
    Collection of util functions
    """

def get_channel_names():
    channels = []
    for task in ["Relaxed1", "Relaxed2", "RelaxedTask1", "RelaxedTask2", "StretchHold", "HoldWeight",
            "DrinkGlas", "CrossArms", "TouchNose", "Entrainment1", "Entrainment2"]:
        for device_location in ["LeftWrist", "RightWrist"]:
            for sensor in ["Acceleration", "Rotation"]:
                for axis in ["X", "Y", "Z"]:
                    channel = f"{task}_{sensor}_{device_location}_{axis}"
                    channels.append(channel)
    return channels
