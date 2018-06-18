import web
import hashlib
import lxml
import time
import os
import requests
from lxml import etree
from tqdm import tqdm
import torch
from torch.autograd import Variable
from torch import nn,optim
import torchvision.transforms as transforms
import json
import os
import pickle
import PIL
import sys
#f_handler=open('out.log', 'w')
#sys.stdout=f_handler
import numpy as np
from functools import partial
import urllib.parse
from time import strftime
from basic_module import BasicModule
from WXBizMsgCrypt import WXBizMsgCrypt
import xml.etree.cElementTree as ET
import logging
logging.basicConfig(filename='log_examp.log',level=logging.DEBUG)
logging.debug('This message should go to the log file')
logging.info('So should this')
logging.warning('And this, too')
urls = ('/wexin','WeixinInterface')
app = web.application(urls,globals())
pickle.load = partial(pickle.load, encoding="latin1")
pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[1.229, 0.224, 0.225])
transformsfunc =  transforms.Compose([
            transforms.Scale(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])									 
def InputImage(imagepath):
	im = PIL.Image.open(imagepath).convert('RGB')
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
resnet = resnet.cuda()

class WeixinInterface:
	def __init__(self):
		self.app_root = os.path.dirname(__file__)
		self.templates_root = os.path.join(self.app_root,'templates')
		self.render = web.template.render(self.templates_root)
	def GET(self):
		data = web.input()
		'''print(data)
		#print("hehehe"+data.signature)
		signature = data.msg_signature
		timestamp = data.timestamp
		nonce = data.nonce
		echostr = data.echostr
		token = 'yizhishijing'
		lis = [token,timestamp,nonce]
		lis.sort()
		#sha1 = hashlib.sha1()
		#map(sha1.update,list)
		#hashcode = sha1.hexigest()
		lis = ''.join(lis)
		if (hashlib.sha1(lis).hexdigest()==signature):
			print("fuck u!")
			return echostr
		#if hashcode == signature:
		#	return echostr'''

		#假设企业在公众平台上设置的参数如下
		sToken = "szh"
		sEncodingAESKey = "y4F0JBPzifdwWHMYOmAlIbzKFvWW2JhD8hOyDVIjY9Y"
		sCorpID = "wwe7a4450fb916941c"
		'''
		------------使用示例一：验证回调URL---------------
		*企业开启回调模式时，企业号会向验证url发送一个get请求 
		假设点击验证时，企业收到类似请求：
		* GET /cgi-bin/wxpush?msg_signature=5c45ff5e21c57e6ad56bac8758b79b1d9ac89fd3&timestamp=1409659589&nonce=263014780&echostr=P9nAzCzyDtyTWESHep1vC5X9xho%2FqYX3Zpb4yKa9SKld1DsH3Iyt3tP3zNdtp%2B4RPcs8TgAE7OaBO%2BFZXvnaqQ%3D%3D 
		* HTTP/1.1 Host: qy.weixin.qq.com

		接收到该请求时，企业应	1.解析出Get请求的参数，包括消息体签名(msg_signature)，时间戳(timestamp)，随机数字串(nonce)以及公众平台推送过来的随机加密字符串(echostr),
		这一步注意作URL解码。
		2.验证消息体签名的正确性 
		3. 解密出echostr原文，将原文当作Get请求的response，返回给公众平台
		第2，3步可以用公众平台提供的库函数VerifyURL来实现。
		'''
		wxcpt=WXBizMsgCrypt(sToken,sEncodingAESKey,sCorpID)
		sVerifyMsgSig=data.msg_signature
		print("msg_signature-->"+sVerifyMsgSig)
		#sVerifyMsgSig=HttpUtils.ParseUrl("msg_signature")
		#sVerifyMsgSig="5c45ff5e21c57e6ad56bac8758b79b1d9ac89fd3"
		sVerifyTimeStamp=data.timestamp
		print("timestamp-->"+sVerifyTimeStamp)
		#sVerifyTimeStamp=HttpUtils.ParseUrl("timestamp")
		#sVerifyTimeStamp="1409659589"
		sVerifyNonce=data.nonce
		print("nonce-->"+sVerifyNonce)
		#sVerifyNonce=HttpUitls.ParseUrl("nonce")
		#sVerifyNonce="263014780"
		sVerifyEchoStr=data.echostr
		print("echostr-->"+sVerifyEchoStr)
		#sVerifyEchoStr=HttpUtils.ParseUrl("echostr")
		#sVerifyEchoStr="P9nAzCzyDtyTWESHep1vC5X9xho/qYX3Zpb4yKa9SKld1DsH3Iyt3tP3zNdtp+4RPcs8TgAE7OaBO+FZXvnaqQ=="
		ret,sEchoStr=wxcpt.VerifyURL(sVerifyMsgSig, sVerifyTimeStamp,sVerifyNonce,sVerifyEchoStr)
		if(ret!=0):
			print("ERR: VerifyURL ret: " + str(ret))
			sys.exit(1)
		else:
			return sEchoStr
		#验证URL成功，将sEchoStr返回给企业号
		#HttpUtils.SetResponse(sEchoStr)

	def POST(self):
		str_xml = web.data().decode()
		print(str_xml)
		xml = etree.fromstring(str_xml)
		print(xml)
		content = xml.find("Content").text
		msgType = xml.find("MsgType").text
		fromUser = xml.find("FromUserName").text
		toUser = xml.find("ToUserName").text
		if msgType == 'image':
			try:
				picurl = xml.find('PicUrl').text
				s = requests.session()
				imagepath = s.get(picurl).content
				inputimg = Variable(InputImage(imagepath))
				inputim = torch.unsqueeze(inputimg,0).cuda()
				output = resnet(inputim)
				_,predicted = torch.max(output.data,1)
				#print(predicted)
				#print("这就是你要的答案")
				return self.render.reply_text(fromUser, toUser, int(time.time()), str(predicted))
			except:
				return self.render.reply_text(fromUser, toUser, int(time.time()),  '识别失败，换张图片试试吧')

if __name__ == "__main__":
	app.run()
