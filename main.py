from dataset_reader.dataset_reader import Reader
import os
path_data = os.path.join(os.getcwd(), "dataset")
geom_data = os.path.join(path_data, "Геометрия")

reader = Reader()
pages = reader.get_array_pages(geom_data)

print(pages)
print(pages[0].image)
print(pages[0].bboxes)
print(pages[0].style)
