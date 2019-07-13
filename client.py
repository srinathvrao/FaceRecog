import requests
import json
import base64


url = 'http://192.168.1.20:8080/image'
# data = {}
# with open('rick.png', mode='rb') as file:
#     img = file.read()
# data['img'] = base64.encodebytes(img).decode("utf-8")
#
# print(json.dumps(data))
data = open('rick.png','rb')
#data = {"eventType": "AAS_PORTAL_START", "data": {"uid": "hfe3hf45huf33545", "aid": "1", "vid": "1"}}
#params = {'sessionKey': '9ebbd0b25760557393a43064a92bae539d962103', 'format': 'xml', 'platformId': 1}
#data = {"image":('rick.png',open('rick.png','rb'))}
params = {'sessionKey':'9129u192849128'}

print(data)

r = requests.post(url, params=params, data=data)
print(r)
