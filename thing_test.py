import requests

response=requests.get("https://www.thingiverse.com/download:6463979")
filename="battery.stl"
with open(filename, 'wb') as f:
    f.write(response.content)
print(f"File '{filename}' downloaded successfully.")