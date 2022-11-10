from scipy import io

loaded = io.loadmat('D:/FinalSet_GIW/Extracted_Data/Extracted_Data/Ball_Catch/Labels/PrIdx_1_TrIdx_2_Lbr_1.mat', squeeze_me=True, struct_as_record=False)
data = loaded['LabelData'][0]

data_path = "D:/FinalSet_GIW/Extracted_Data/Extracted_Data/Ball_Catch/ProcessData_cleaned/PrIdx_1_TrIdx_2.mat"

loaded = io.loadmat(data_path, squeeze_me=True, struct_as_record=False)
data = loaded['ProcessData'].ETG
data = loaded['ProcessData'].PrIdx  #
data = loaded['ProcessData'].TrIdx  #

scene_frames = data.SceneFrameNo
