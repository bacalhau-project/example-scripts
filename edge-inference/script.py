import jetson_emulator.inference as inference
import jetson_emulator.utils as utils

# load the recognition network
net = inference.imageNet("googlenet")
for x in range(1,6):
	# emulator API to generate sample images for imageNet
	filename = net.emulatorGetImageFile()      
	img = utils.loadImage(filename) 
	class_idx, confidence = net.Classify(img)            
	class_desc = net.GetClassDesc(class_idx)            
	print("image "+str(x)+" is recognized as '{:s}' (class #{:d}) with {:f}% confidence".
	format(class_desc, class_idx, confidence*100))