import xml.etree.ElementTree as ET

# XML 파일 경로
xml_file_path = 'sunnyday_assc_gt.xml'

output_txt_file_path = 'output.txt'

# XML 파일 파싱
tree = ET.parse(xml_file_path)
root = tree.getroot()

# 변환된 데이터를 저장할 리스트
converted_data = []

# XML 데이터를 파싱하여 원하는 형식으로 변환
for frame in root.findall('frame'):
    frame_number = int(frame.get('number'))
    frame_number = (frame_number // 10) * 10  # 프레임 번호를 10의 배수로 변경
    for obj in frame.find('objectlist').findall('object'):
        obj_id = int(obj.get('id'))
        box = obj.find('box')
        xc = float(box.get('xc'))
        yc = float(box.get('yc'))
        converted_data.append(f"{frame_number}\t{obj_id}\t{xc:.2f}\t{yc:.2f}")

# 변환된 데이터를 텍스트 파일로 저장
with open(output_txt_file_path, 'w') as file:
    for data in converted_data:
        file.write(data + '\n')

print(f"Data has been successfully written to {output_txt_file_path}")