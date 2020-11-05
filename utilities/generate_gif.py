from PIL import Image
import glob

'''
This is a simple script to generate a gif by passing in a path of frames in order. 
Pass in a path to retrieve the images and save the file with appropriate name.
'''
# Create the frames
frames = []
imgs = glob.glob("test\\vid4\\calendar\\Proposed_Mean_Loss_20200817_Proposed_Loss_20200819\\*.png")
for i in imgs:
    print(i)
    new_frame = Image.open(i)
    frames.append(new_frame)

# Save into a GIF file that loops forever
frames[0].save('report/calendar_mean.gif', format='GIF',
               append_images=frames[5:-5],
               save_all=True,
               duration=300, loop=0)
