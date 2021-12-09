import os

folders = os.listdir()

index = 0
for folder in folders:
	#print(folder)
	#print("ffmpeg -i "+folder+"/"+folder[:-4]+"_ambient.avi -f image2 -start_number 0 "+folder+"/images_ambi/frame_%06d.PNG")
	#paddedIndex = str(index).zfill(5)
	os.system("ffmpeg -i "+folder+"/"+folder[:-4]+"_ambient.avi -f image2 -start_number 0 "+folder+"/images_ambi/frame_%06d.PNG")
	os.system("ffmpeg -i "+folder+"/"+folder[:-4]+"_range.avi -f image2 -start_number 0 "+folder+"/images_range/frame_%06d.PNG")
	os.system("ffmpeg -i "+folder+"/"+folder[:-4]+"_intensity.avi -f image2 -start_number 0 "+folder+"/images_int/frame_%06d.PNG")
	#index += 1