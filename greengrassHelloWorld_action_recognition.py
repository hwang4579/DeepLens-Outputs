#
# Copyright Amazon AWS DeepLens, 2017
#

# greengrassInfiniteInfer.py
# Runs GPU model inference on a video stream infinitely, and
# publishes a message to topic 'infinite/infer' periodically.
# The script is launched within a Greengrass core.
# If function aborts, it will restart after 15 seconds.
# Since the function is long-lived, it will run forever 
# when deployed to a Greengrass core. The handler will NOT 
# be invoked in our example since we are executing an infinite loop.

import os
import greengrasssdk
from threading import Timer
import time
import awscam
import cv2
import boto3
from threading import Thread

# Creating a greengrass core sdk client
client = greengrasssdk.client('iot-data')

# The information exchanged between IoT and clould has 
# a topic and a message body.
# This is the topic that this code uses to send messages to cloud
iotTopic = '$aws/things/{}/infer'.format(os.environ['AWS_IOT_THING_NAME'])

ret, frame = awscam.getLastFrame()
ret,jpeg = cv2.imencode('.jpg', frame) 
Write_To_FIFO = True
class FIFO_Thread(Thread):
    def __init__(self):
        ''' Constructor. '''
        Thread.__init__(self)
 
    def run(self):
        fifo_path = "/tmp/results.mjpeg"
        if not os.path.exists(fifo_path):
            os.mkfifo(fifo_path)
        f = open(fifo_path,'w')
        client.publish(topic=iotTopic, payload="Opened Pipe")
        while Write_To_FIFO:
            try:
                f.write(jpeg.tobytes())
            except IOError as e:
                continue  
            
# Function to write to S3
# The function is creating an S3 client every time to use temporary credentials
# from the GG session over TES
def write_image_to_s3(img):
    client.publish(topic=iotTopic, payload='write_image_to_s3')
    s3_client = boto3.client('s3')
    file_name = 'DeepLens/image-'+time.strftime("%Y%m%d-%H%M%S")+'.jpg'

    # You can contorl the size and quality of the image
    encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),90]
    _, jpg_data = cv2.imencode('.jpg', img, encode_param)
    response = s3_client.put_object(ACL='public-read', Body=jpg_data.tostring(),Bucket='deeplens-sagemaker-heng',Key=file_name)
    image_url = 'https://s3.amazonaws.com/deeplens-sagemaker-heng/'+file_name
    return image_url

# Adding top n bounding boxes results
def apply_bounding_box(img, topn_results):
        '''
        cv2.rectangle(img, (x1, y1), (x2, y2), RGB(255,0,0), 2)
        x1,y1 ------
        |          |
        --------x2,y2
        '''
        for obj in topn_results:
            class_id = obj['label']
            class_prob = obj['prob']
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.rectangle(img, (int(obj['xmin']), int(obj['ymin'])), (int(obj['xmax']), int(obj['ymax'])), (255,0,0), 2)
        return img

# When deployed to a Greengrass core, this code will be executed immediately
# as a long-lived lambda function.

def greengrass_infinite_infer_run():
    try:
        modelPath = "/opt/awscam/artifacts/mxnet_action_recognition_v2.0_FP16_FUSED.xml"
        modelType = "classification"
        outMap = {0: 'applyeyemakeup', 1: 'applylipstick', 2: 'archery', 3: 'basketball', 4: 'benchpress', 5: 'biking', 6: 'billiards', 7: 'blowdryhair', 8: 'blowingcandles', 9: 'bowling', 10: 'brushingteeth', 11: 'cuttinginkitchen', 12: 'drumming', 13: 'haircut', 14: 'hammering', 15: 'handstandwalking', 16: 'headmassage', 17: 'horseriding', 18: 'hulahoop', 19: 'jugglingballs', 20: 'jumprope', 21: 'jumpingjack', 22: 'lunges', 23: 'nunchucks', 24: 'playingcello', 25: 'playingflute', 26: 'playingguitar', 27: 'playingpiano', 28: 'playingsitar', 29: 'playingviolin', 30: 'pushups', 31: 'shavingbeard', 32: 'skiing', 33: 'typing', 34: 'walkingwithdog', 35: 'writingonboard', 36: 'yoyo'}
        # Send a starting message to IoT console
        client.publish(topic=iotTopic, payload="Action recognition starts now")
        results_thread = FIFO_Thread()
        results_thread.start()

        # Load model to GPU (use {"GPU": 0} for CPU)
        mcfg = {"GPU": 1}
        model = awscam.Model(modelPath, mcfg)
        client.publish(topic=iotTopic, payload="Model loaded")

        doInfer = True
        while doInfer:
            # Get a frame from the video stream
            ret, frame = awscam.getLastFrame()

            # Raise an exception if failing to get a frame
            if ret == False:
                raise Exception("Failed to get frame from the stream")
            
            # Resize frame to fit model input requirement
            frameResize = cv2.resize(frame, (224, 224))

            # Run model inference on the resized frame
            inferOutput = model.doInference(frameResize)

            outputProcessed = model.parseResult(modelType, inferOutput)

            # Get top 5 results with highest probiblities
            topFive = outputProcessed[modelType][0:5]
            
            # put message
            msg = "{"
            probNum = 0 
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, outMap[topFive[0]["label"]], (0,22), font, 1, (255, 165, 20), 4)
            for obj in topFive:
                if probNum == 4: 
                    msg += '"{}": {:.2f}'.format(outMap[obj["label"]], obj["prob"])
                else:
                    msg += '"{}": {:.2f},'.format(outMap[obj["label"]], obj["prob"])
                probNum += 1
            msg += "}"
            client.publish(topic=iotTopic, payload=msg)

            global jpeg
            ret,jpeg = cv2.imencode('.jpg', frame)
            
            results = model.parseResult('ssd',inferOutput)
            tops = results['ssd'][0:5]
            frame_with_bb = apply_bounding_box(frameResize,tops)
            # Upload to S3 to allow viewing the image in the browser
            image_url = write_image_to_s3(frame_with_bb)
            client.publish(topic=iotTopic, payload='{{"img":"{}"}}'.format(image_url))
        
    except Exception as e:
        msg = "Test failed: " + str(e)
        client.publish(topic=iotTopic, payload=msg)

    
    # Asynchronously schedule this function to be run again in 15 seconds
    Timer(15, greengrass_infinite_infer_run).start()

# Execute the function above
greengrass_infinite_infer_run()

# This is a dummy handler and will not be invoked
# Instead the code above will be executed in an infinite loop for our example
def function_handler(event, context):
    return

