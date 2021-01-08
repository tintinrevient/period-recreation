from quickdraw import QuickDrawData

# one drawing
# qd = QuickDrawData()
# yoga = qd.get_drawing("yoga")
#
# print(yoga.name)
# print(yoga.key_id)
# print(yoga.countrycode)
# print(yoga.recognized)
# print(yoga.timestamp)
# print(yoga.no_of_strokes)
# print(yoga.image_data)
# print(yoga.strokes)
#
# yoga.image.save("./data/my_yoga.gif")

# multiple drawings
from quickdraw import QuickDrawDataGroup
import os

yogas = QuickDrawDataGroup("yoga", max_drawings=None)
print(yogas.drawing_count)

yogas = QuickDrawDataGroup("yoga", max_drawings=5000)
idx = 0
for drawing in yogas.drawings:
    idx += 1
    filepath = os.path.join('data', 'my_yoga_' + str(idx) + '.gif')
    drawing.image.save(filepath)