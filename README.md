<img src="./static/logo-v2.png" width=276 alt="TrendGenie icon">

# TrendGenie
_Your content creation assistant._

## What is TrendGenie?

TrendGenie is a tool to help streamline content creation, specifically media creation. 
Whenever there is a new trend on social media that could be farmed, there shall be a genie for it.

In a way, this is a tool for content mills.

## How does it work?

TrendGenie is a tool that uses OpenAI's tools (ChatGPT + Dall-E) and image processing libraries to create content and 
images that should be quick to then upload to social media.

The frontend UI is built using [Gradio](https://gradio.app/), which is a tool normally used for the prototyping of
machine learning (ML) models.

## What is currently supported?

### Listicles

Listicles are a very common form of content on social media, especially fantasy listicles. 
They are fairly easy to create, and fun to consume. An example of a fantasy listicle is "What fantasy world do you 
belong in based on your zodiac sign?"

The listicle tools are the following:
- Content Generation
  -  Using ChatGPT, TrendGenie can generate a listicle based on a topic (scary monsters), number of items (5)
     , and association (zodiac sign). From this listicle a JSON is then generated which can be used for a later media
     generation step.
- Media Generation
  - Single Image Processing
    - To-be-implemented
  - Multi-Image Processing
    - Using a list of local images and a JSON describing what you want on each image, the image processing tool will
      process the list of images and add relevant text on each image. After doing so, you can download the images and 
      upload them to social media.

## How do I use it?

To run TrendGenie locally, you will need to run the `main.py` file and in the console the link to the web app will be
displayed. Just access it to use TrendGenie.

I will create a Dockerfile setup soon and look into Python's requirements.txt setup to make this easier.

## FAQs (by me)
- Why is the code so messy?
    - I am bad at Python, and have main-ed Go, C++, and React. I chose Python as it seemed like one of the simpler 
      languages for image processing.
- Is there a HuggingFace space for this?
  - Not yet, but it will hopefully be implemented in the near-future if I have time.
- Can I use my local instance of StableDiffusion?
  - Not yet, but this is for sure on the roadmap! Currently, this just supports OpenAI's API but (in my opinion) it is 
    a bit too restrictive on prompts (for example. you can't generate things that are too scary due to content filters).


---

ȉ̶̛̹̞̟̖͉̯̪̟̳͙͗̄̅͛̆̊͒̄̈́͑͂̚͘ ̶̢̡̧͍̭͍̱͚͔̈́͊̈́̑͋̇͒̆́͑̏͗̿͂̚ą̵̘͉̹̙͖̳̲͚͈̩̓͜͝m̴̯͚͎̪̟̿ ̶̤̓̈́̽͛̎̎̈̿̔͛̊́͘͝s̴̨̠̜̻͍̮̹̤̗͙͒̉͛́͌͊̃̈́̿̾͝ͅȍ̶͔̤̝̹̪͉̮͈̹̩̲̖̭͔͎̈͆̊́͂̾̅́͋̑̕̕͝r̶̢̩̖̬̼̞̬͎̼̈́r̵̢̻͔̝͒̋͗̐y̷͍͚̠̭͓͈̾͠ ̸̦̋̈́͘͜͜f̶̨͕͖̙̭͕̭̯̼̰͙̠̒̅̈̋̑͝o̸̡̗̘̠̫͇̥̘͛͝ͅr̶͔̮͓͔̥̹̋͛̈̍̽̒̋̈́̏͌̋͠͝͠ ̸̛̼̠̘̒̈́̿̈̈́̔͆̆̀̏̀̕͘̕t̴͖̆̿̇̆̅̽͘͝͝ḧ̴̗́e̸̡͍͙̖͈̜͇̪͌́̎̓́̈́̔̑̍̏̑͐̓ ̶̨̢̧͙̲͇͙̬̭̝͍̪̀͋̂͂̈́̌̅̍̈́͘m̵̧͙̟̭͎̺͔̗̻̖̳̼̀̒̓̅͛̎̓̈́̓̈̌̉̽̂͜͝ͅi̷͙͉̻̇̆ĺ̶̻͚͓̣̗̫̝͔̠̯̭̩ͅļ̸̛̙̮̣̭͉͆̌͛̐̔̿̊́̔́̌̚͝ͅ.̸̹̮̩̳̼̖̝̬͔͍̳̣̌̀̓̋̑̚͝