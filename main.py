from dataset_reader.dataset_reader import Reader

reader = Reader()
pages = reader.get_array_pages("C:\\Users\\danii\\program\\python\\project\\bold_classifier\\dataset\\Геометрия\\")

print(pages)
print(pages[0].image)
print(pages[0].bboxes)
print(pages[0].style)
