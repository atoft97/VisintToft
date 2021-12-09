import json

def insertVideo(data, i):
	data['videos'] = [
		{
			"id": i,
			"name": "video"+str(i)+"AIR.mp4"
		}
		]
	return (data)

def addVideoIDtoImages(data, i):
	counter = 0
	for image in data['images']:
		image['video_id'] = i
		image['frame_id'] = counter
		counter += 1

	return(data)

def addVideoIDtoAnnotations(data, i):
	for image in data['annotations']:
		image['video_id'] = i
		image['is_vid_train_frame'] = True

	return(data)

for i in range(9,10):

	f = open('Video0000'+str(i)+'_Ano/annotations/instances_default.json')
	data = json.load(f)
	data = insertVideo(data, i)
	data = addVideoIDtoImages(data, i)
	data = addVideoIDtoAnnotations(data, i)



	f.close()

	out_file = open("Video0000"+str(i)+"_Ano/train"+str(i)+".json", "w")
	json.dump(data, out_file)


