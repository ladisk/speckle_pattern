Generate print-ready images of a random speckle pattern for DIC applications.

#### Installation

This package is hosted on PyPI. Install it using `pip`:

```pip install speckle_pattern```


#### Example speckle pattern

```python
from speckle_pattern import generate_and_save

image_height = 50 # mm
image_width = 100 # mm
speckle_diameter = 3 # mm
dpi = 200
save_path = 'test.jpg'

generate_and_save(image_height, image_width, dpi, speckle_diameter, save_path)
```

<p align='center'><img src='example.jpg' width=400 alt='random speckle'/></p>


#### Example line pattern

```python
from speckle_pattern import generate_lines

image_height = 50 # mm
image_width = 100 # mm
line_width = 5 # mm
orientation = 'vertical'
dpi = 200
save_path = f'example_lines_{orientation}.jpg'

generate_lines(image_height, image_width, dpi, line_width, save_path)
```

<p align='center'><img src='example_lines_vertical.jpg' width=400 alt='random speckle'/></p>


### Authors

- [Domen Gorjup](http://ladisk.si/?what=incfl&flnm=gorjup.php)
- [Janko Slavič](http://ladisk.si/?what=incfl&flnm=slavic.php)
- [Miha Boltežar](http://ladisk.si/?what=incfl&flnm=boltezar.php)