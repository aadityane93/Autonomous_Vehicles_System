copy the data set from the robot using ssh with the command replace the road_following_A with the appropriate path the images are. eg road_following_B. replace the ip with the ip of the jetracer.

```
scp -r jetson@192.168.137.148:jetracer/notebooks/road_following_A/apex ./road_following_A/apex
```

Place the data set in
```
road_following_A\apex
```

run the file clean_broken_images.py to clean broken images the names of the deleted broken images are saved to the a new txt file.
```
python clean_broken_images.py
```

in (train_road_following.py) you can change the batch size, the number of epochs and the model name then run it.

