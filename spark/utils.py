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

def channel_wise_boss(time_data, word_size = 2, window_size = 30, window_step = 2):
    boss = BOSS(word_size = word_size, window_size = window_size, window_step = window_step)
    boss_output = []
    for c_idx in range(time_data.shape[1]):
        c_data = time_data[:,c_idx,:]
        c_feat = boss.fit(c_data)
        print(c_feat)
        boss_output.append(c_feat)
    return boss_output

def boss_transform_data(boss_output,X):
    X_boss = []
    for s_idx in range(X.shape[0]):
        for c_idx in range(X.shape[1]):
            channel_boss= boss_output[c_idx]
            boss_features = channel_boss.transform(X[s_idx,c_idx,:].reshape(1,-1))
            X_boss.append(boss_features.toarray())
    print(len(boss_features.toarray()))
    n_feat = len(boss_features.toarray()[0])
    X_boss_reshape = np.array(X_boss).reshape(len(X),X.shape[1]*n_feat)
    return X_boss_reshape
