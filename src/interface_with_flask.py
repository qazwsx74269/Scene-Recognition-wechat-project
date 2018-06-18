# -*- encoding:utf-8 -*-
from flask import Flask,request,make_response
from WXBizMsgCrypt import WXBizMsgCrypt
import time
import os
import torch
from torch.autograd import Variable
from torch import nn,optim
import torchvision.transforms as transforms
import os
import pickle
import PIL
import sys
import requests
from scipy import misc
#f_handler=open('out.log', 'w')
#sys.stdout=f_handler
import numpy as np
from functools import partial
from time import strftime
from basic_module import BasicModule
from WXBizMsgCrypt import WXBizMsgCrypt
import xml.etree.cElementTree as ET
import pandas as pd
import urllib
from io import BytesIO
clas_file = './scene_classes.csv'
a=pd.read_csv(clas_file,header=None)
pickle.load = partial(pickle.load)
pickle.Unpickler = partial(pickle.Unpickler)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[1.229, 0.224, 0.225])
transformsfunc =  transforms.Compose([
            transforms.Scale(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])									 
def InputImage(imagepath):
	#print "qunimade"
	im = PIL.Image.open(BytesIO(imagepath.content)).convert('RGB')
	#print imagepath
	#im = misc.imread(imagepath)
	print("made")
	return transformsfunc(im)

class ResNet(BasicModule):
    def __init__(self, model,opt=None, feature_dim=2048, name='resnet'):
        super(ResNet, self).__init__(opt)
        self.model_name = name
		# model.avgpool_1a = nn.AdaptiveAvgPool2d(1)
		# del model.classif
		# model.classif = lambda x: x
        model.avgpool = nn.AdaptiveAvgPool2d(1)
        del model.fc
        model.fc = lambda x: x
        self.features = model
        self.classifier = nn.Linear(feature_dim, 80)

    def forward(self, x):
        features = self.features(x)
        return self.classifier(features)

model = torch.load('/home1/szh/scene-baseline-master/checkpoints/whole_resnet50_places365.pth.tar', map_location=lambda storage, loc: storage, pickle_module=pickle)
resnet = ResNet(model)
resnet.load_state_dict(torch.load("/home1/szh/scene-baseline-master/checkpoints/res_365_1203_0055_0.988225053566")['d'])
resnet.eval()
print "模型加载完成。。。。"
app = Flask(__name__)
@app.route('/wexin',methods=['GET','POST'])
def wechat():
	sToken = "szh"
	sEncodingAESKey = "y4F0JBPzifdwWHMYOmAlIbzKFvWW2JhD8hOyDVIjY9Y"
	sCorpID = "wwe7a4450fb916941c"
	wxcpt=WXBizMsgCrypt(sToken,sEncodingAESKey,sCorpID)
	if request.method=='GET':
		data=request.args
		sVerifyMsgSig=data.get('msg_signature','')
		sVerifyTimeStamp=data.get('timestamp','')
		sVerifyNonce =data.get('nonce','')
		sVerifyEchoStr=data.get('echostr','')
		ret,sEchoStr=wxcpt.VerifyURL(sVerifyMsgSig, sVerifyTimeStamp,sVerifyNonce,sVerifyEchoStr)
		if(ret!=0):
			print("ERR: VerifyURL ret: " + str(ret))
			sys.exit(1)
		else:
			return sEchoStr
	else:
		data=request.args
		rec = request.stream.read()
		sReqMsgSig=data.get('msg_signature','')
		sReqTimeStamp=data.get('timestamp','')
		sReqNonce =data.get('nonce','')
		ret,sMsg=wxcpt.DecryptMsg(rec, sReqMsgSig, sReqTimeStamp, sReqNonce)
		if( ret!=0 ):
			print("ERR: DecryptMsg ret: " + str(ret))
			sys.exit(1)
		
		xml = ET.fromstring(sMsg)
		#content = xml.find("Content").text
		msgType = xml.find("MsgType").text
		fromUser = xml.find("FromUserName").text
		toUser = xml.find("ToUserName").text
		xml_rep = "<xml><ToUserName><![CDATA[%s]]></ToUserName><FromUserName><![CDATA[%s]]></FromUserName><CreateTime>%s</CreateTime><MsgType><![CDATA[text]]></MsgType><Content><![CDATA[%s]]></Content><FuncFlag>0</FuncFlag></xml>"
		print msgType
		if msgType == 'image':
			try:
				picurl = xml.find('PicUrl').text
				print picurl
				#urllib.urlretrieve(picurl,'%s.jpg' %("test"))
				#s = requests.session()
				#imagepath = "./test.jpg"
				#imagepath = s.get(picurl).content
				#print type(imagepath)
				#inputimg = Variable(InputImage(imagepath))
				pic = requests.get(picurl)
				inputimg = Variable(InputImage(pic))
				inputim = torch.unsqueeze(inputimg,0)
				output = resnet(inputim)
				_,predicted = torch.max(output.data,1)
				print str(a[1][predicted]+"/"+a[2][predicted]) 
				print "这就是你要的答案"
				sRespData = xml_rep % (fromUser,toUser,str(int(time.time())),str(a[1][predicted]+"/"+a[2][predicted]))
				print 1
				ret,sEncryptMsg=wxcpt.EncryptMsg(sRespData, sReqNonce, sReqTimeStamp)
				print 2
				if( ret!=0 ):
					print("ERR: EncryptMsg ret: " + str(ret))
					sys.exit(1)
				print 3
				return make_response(sEncryptMsg)
			except:
				sRespData = xml_rep % (fromUser,toUser,str(int(time.time())),'识别失败，换张图片试试吧')
				return make_response(sRespData)
		else:
			sRespData = xml_rep % (fromUser,toUser,str(int(time.time())),"hello")
			print(sRespData)
			ret,sEncryptMsg=wxcpt.EncryptMsg(sRespData, sReqNonce, sReqTimeStamp)
			if( ret!=0 ):
				print("ERR: EncryptMsg ret: " + str(ret))
				sys.exit(1)
			return make_response(sEncryptMsg)

'''if __name__ == '__main__':
    b = True
    print "start"
    while(b):
        try:
            app.run(port=50012,threaded=True)
            b = False
        except:
            print "haha"
            k=0'''