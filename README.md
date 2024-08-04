```sh
 ________ ________  ____    ____    ________  ________  ________  ________  ________ 
|________||   _   ||    \  /    |  |   _   ||        ||   _   ||   _   | |        |
    |    ||  |_|  ||  |  \/  |  |  |  |_|  ||  .--.  ||  |_|  ||  |_|  | |  .--.  |
    |    ||       ||  |      |  |  |       ||  |  |  ||       ||       | |  |  |  |
 ___|    ||   _   ||  |      |  |  |   _   ||  '--'  ||   _   ||   _   | |  '--'  |
|        ||  | |  ||  |      |  |  |  | |  ||        ||  | |  ||  | |  | |        |
|________||__| |__||__|      |__|  |__| |__||________||__| |__||__| |__| |________|
```

## Inspiration
If you've ever seen a YouTube video of someone composing music from scratch, playing piano and guitar in their homes, or out busking on the streets, you may have thought: "I want to play an instrument like that too." For those of us fortunate enough to pursue our musical fantasies, this problem is solved as simply as buying a new instrument. However, there are others who do not have the capacity to own an instrument on a whim. We wanted to change that.

## What it does
JamBoard is a tool similar to a DJ's soundboard that allows you to create music by drawing shapes that represent drums, piano, saxophone, and more on a piece of paper. When you connect your device (i.e., budget Android phone), you can tap on the various shapes which creates pitched musical notes. The sounds are consistent and so you can effectively create musical pieces with it (see our attached video!)

Instruments supported: piano, saxophone, drums, and kazoo (and more can be automatically generating by providing just one sound file)

## How we built it
We built JamBoard using what is essentially pure math. 
- On startup 'calibration' occurs, we run the webcam through a series of processes to turn it into a black and white shape grid. We then essentially count the number of contours and how round the object is to determine it's shape, area, color and exact location - no AI tomfoolery.
- Two webcams are set up, one with a bird's eye view, and one with a table's eye view. They create a constant feed of video frames, which we use baselines from the calibration step to determine when the user is touching a certain shape.
- When a shape is touched, we use the corresponding instrument's singular sound file to generate a pitched note based on the colour and size of the shape.  
- When you drag your finger from one shape to another, they interpolate (blend) together and create a smooth sliding sound.

### TECHNICAL DIFFICULTY
Since working with similar image processing pipelines are a relatively common route for hackathons, we wanted to challenge ourselves by implementing less common features. These include: 
- Not hardcoding musical notes; Each pitch that you hear from your instrument-shapes are generated during runtime. If you check our GitHub repository, there's no folder filled with C4.mp3, D5.mp3, etc. This was challenging because there was no one-size-fits-all solution. We had to jump through a lot of hoops that were not necessarily well-documented.
- Note Interpolation Feature; One cool idea we had was the concept of a slider: a way for you to press on one note, drag your finger to another note, and have the sounds in between generated smoothly. This was especially hard to implement and we had to break out our iPad and Desmos graphing calculator to do some tough math.
- Writing all of the image processing ourself: Instead of using some basic AI model or calling the Gemini API, we have our own custom pure math solution. We wrote this using the OpenCV library, yet in our case we were simply using it as a more specialized NumPy.
- We don't rely on external APIs. 

## DESIGN
Instead of HTML and CSS, our design processes consisted of us analyzing what kind of musical experience seems "most human."  We decided on our current system because it's easy for JamBoarders to differentiate between their notes, and you do not need to know how to draw.

When you create a shape, the colour represents the musical notes, the size represents the octave, and the shape represents the instrument. Our Human-Computer Interaction's sophistication lies in its simplicity, as that is how we believe systems should be designed.

## Challenges we ran into
- The action of tapping a shape was much harder to implement than we expected. We initially had the novel idea of using shadows and its relative proportions to tell if a finger is close to the table. While we got a working prototype, any change in the building's lighting destroyed all consistency.
- Managing a limited number of webcams was tough during development. We had three webcams that basically were hot-potatoed around the team as we took turns developing features.

## What we learned
- We learned that music hacks are actually really interesting. Throughout our debugging process, the air was always filled with sounds. We also remade a new version of the C major scale (ours is better)

## Thanks for reading!
Feel free to ask any questions at our booth on August 4, 2024
