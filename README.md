# video_to_visual_podcast

## Overview

An app that converts YouTube videos into commentary podcasts between two talking avatars discussing the video content.

## Usage

##### Note : The app doesn't work if one of the potrait images of the speaker is detected to be of a celebrity or under the age of 18.

You can generate some test potrait images here: 

- https://thispersondoesnotexist.com/



### Input: Youtube video url and potrait images for the speakers.


### Parameters

- `url`: 
  The URL of the YouTube video to be processed.

- `name1`:
  Name of speaker one in the conversation.

- `voice1`:
   Voice of speaker one in the conversation.

- `potrait_image1`:
  Image avatar of speaker one in the conversation.

- `name2`:
  Name of speaker one in the conversation.

- `voice2`:
   Voice of speaker two in the conversation.

- `potrait_image2`:
  Image avatar of speaker two in the conversation.

- `max_summary_length`: 
  The maximum length of the summary for the video that gets talked about.

## How it works
Upon receiving a youtube video link, the app first downloads the video , extracts the audio and then runs a speech-to-text process to transcribe the content. It then processes the transcript to generate the summary, using LLMs. 

The summary is further transformed into a conversational dialogue between two individuals, based on the provided names, using LLMs.

Finally, the generated conversation is converted into speech, with talking avatars assigned to each speaker based on their dialogue.

## Additionaly you need the following in your ENV:

- AZURE_OPENAI_API_KEY : AZURE_OPENAI_API_KEY is the API key of your deployed model.

- AZURE_API_VERSION : AZURE_API_VERSION of your deployed endpoint.

- AZURE_OPENAI_ENDPOINT : AZURE_OPENAI_ENDPOINT is the endpoint of your deployed model

- AZURE_DEPLOYMENT_NAME : AZURE_DEPLOYMENT_NAME of your deployed model.

### Example :

 For finding the last three in your AZURE AI studio there might be something like this:

`https://xyz.openai.azure.com/openai/deployments/abc/chat/completions?api-version=2024-02-15-preview`

where `https://xyz.openai.azure.com` is your AZURE_OPENAI_ENDPOINT.

where `abc` is your AZURE_DEPLOYMENT_NAME.

where `2024-02-15-preview` is your AZURE_API_VERSION.



    