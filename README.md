# AWS DeepComposer

AWS DeepComposer provides a creative, hands-on experience for learning generative AI and machine learning. With generative AI, one of the biggest recent advancements in artificial intelligence, you can create a new dataset based on a training dataset. With AWS DeepComposer, you can experiment with different generative AI architectures and models in a musical setting by creating and transforming musical inputs and accompaniments.
<img src="https://github.com/og121public/aws-deepcomposer/blob/main/images/2.jpg?raw=true" height="496">

Regardless of your experience with machine learning (ML) or music, you can use AWS DeepComposer to develop a working knowledge of generative AI. AWS DeepComposer includes learning capsules, sample code, and training data to help you understand and use generative AI models.
<img src="https://github.com/og121public/aws-deepcomposer/blob/main/images/5.jpg?raw=true" height="496">

To get started with AWS DeepComposer, start the AWS DeepComposer Music studio, choose one of the sample melodies, and choose a pretrained model. After you generate a composition, you can change the instruments, download your new composition, and share it with friends on SoundCloud.

To flex your creativity, record a custom melody using either the console-based keyboard or the AWS DeepComposer keyboard. To dive deeper, start training custom models using training data provided by AWS. Want more? Learn how to create your own generative adversarial networks (GANs) by using the examples in Jupyter notebooks for SageMaker.
<img src="https://github.com/og121public/aws-deepcomposer/blob/main/images/4.gif?raw=true" height="891">

# AWS DeepComposer concepts

AWS DeepComposer builds on the following concepts and uses the following terminology.

### autoregressive convolutional neural network (AR-CNN)

A generative AI technique that edits your input melody. This technique uses a U-Net architecture and was trained on a collection of chorales by Johann Sebastian Bach. The AR-CNN technique works by detecting notes that sound missing or out place based on the training dataset. The identified notes are then replaced with notes that the model thinks would fit the distribution of notes learned during training.

### AutoregressiveCNN Bach

A pretrained autoregressive convolutional neural network (AR-CNN) model that is available in the AWS DeepComposer music studio. This model has been trained on a dataset containing only Bach chorales. It uses the U-net architecture.

### AWS DeepComposer keyboard

Also known as the AWS DeepComposer physical keyboard or hardware keyboard. You can connect, or link, the keyboard to a computer that has access to the AWS DeepComposer console. You can use the linked keyboard to play and record short melodies that are fewer than eight bars. Then use a recorded melody with a supported AWS DeepComposer generative AI technique.

### compositions

Sequences of notes, melodies, harmonies, and rhythms that make up a musical work.

In AWS DeepComposer, a composition also represents a saved piece of music. Each time you use a generative technique to perform inference, your output is automatically saved as a composition.

### convolutional neural network (CNN)

A type of neural network commonly used for image-recognition and video-recognition tasks. A CNN filters and summarizes the data during training to learn patterns. This mathematical process is called _convolution_.

### decoder

A process that takes an input vector, commonly a feature matrix, and transforms the vector into an output. A decoder can be used with an encoder. AWS DeepComposer uses an encoder-decoder network for some generative AI tasks.

### discriminator

A classifier model, one part of a generative adversarial network (GAN). The discriminator classifies data as real or as generated. In AWS DeepComposer, this model tries to determine if a generated piano roll looks like a real image or an artificially created image. The discriminator is trained on real data.

### drum in pattern

A ratio of the total number of notes in a drum track to a predetermined beat popular in 4/4 time.

### empty bar rate

The ratio of empty bars to the total number of bars.

### encoder

A process that translates data from one format to another. AWS DeepComposer uses multiple encoders. For example, it uses an encoder to translate an input MIDI file into a piano roll image. Often, an encoder is used with a decoder.

### epoch

One complete pass through the training dataset by the neural network. For example, if you have 10,000 music tracks in the training dataset, one epoch represents one pass through all 10,000 tracks. The number of epochs required for a model to converge varies based on the training data. An iterative algorithm converges when the loss function stabilizes.

### generative adversarial network (GAN)

Two neural networks that consist of a _generator_ and a _discriminator_. In AWS DeepComposer, the generator learns to compose music that sounds as realistic as possible with feedback from the discriminator. The discriminator treats the generator's output as sounding as unrealistic as possible while holding the input training sample as the ground truth. Training proceeds, with the generator searching for its network weights by minimizing the chances that its generations differ from the training samples. The discriminator searches for the weights of both networks by minimizing the chances it misjudges the training samples as realistic compositions and the generator's outputs as unrealistic compositions. The efforts made by both the generator and the discriminator push the generator's network weights in opposite directions. This is the essence of the adversarial role that the discriminator takes against the generator. Learning proceeds until equilibrium is reached, where the generator improves its output to such a degree that the discriminator can no longer distinguish between the generated composition and training samples.

### hyperparameters

Algorithm-dependent variables that you can control. You can tune hyperparameters to find the best fit for a specific problem that you are trying to model.

### inference

Predictions generated by a trained model.

### in scale ratio

The ratio of the average number of notes in a bar of music that are in the key of C, to the total number of notes in a bar of music.

### learning rate

A hyperparameter used for training neural networks. The learning rate controls how much the weights and biases are updated during training.

### loss function

Evaluates how effective an algorithm is at modeling the data. For example, if a model consistently predicts values that are very different from the actual values, it returns a large loss. Depending on the training algorithm, more than one loss function might be used.

### model

The final output created while training with a machine learning algorithm. The algorithm used to train the model finds patterns in the training data uses those patterns to map the input data attributes to the target (the answer that you want to predict). The algorithm outputs a machine learning model that captures these patterns.

### MuseGAN

A generative adversarial network (GAN) architecture built specifically to generate music. Like other GANs, MuseGAN is made of both a discriminator and a generator that use a CNN. The MuseGAN architecture is available in AWS DeepComposer. To learn more about the MuseGAN architecture, see github.com/salu133445/musegan

### Music studio

A component of the AWS DeepComposer console that provides access to a console-based keyboard for playing, recording, and composing music. With Music studio, you can try out AWS DeepComposer before purchasing an AWS DeepComposer physical keyboard. You don't need to train a model to use Music studio.

### neural network

Also known as an artificial neural network. A collection of connected units or nodes that are used to build an information model based on biological systems. Each node is called an artificial neuron. An artificial neuron mimics a biological neuron in that it receives an input (stimulus), becomes activated if the input signal is strong enough (activation), and produces an output predicated on the input and activation. Neural networks are widely used in machine learning because an artificial neural network can serve as a general-purpose approximation to any function.

### pitch

The frequency of a sound, often judged to be high or low. In the AWS DeepComposer Music studio, you can adjust the pitch of a sample, imported, or custom recorded input melody.

### pitches used

A metric that captures the average number of notes in each bar.

### polyphonic rate

The ratio of the number of time steps where the number of pitches being played is greater than the **threshold** number of allowable time steps.

### Rhythm assist

An AWS DeepComposer function that automatically corrects the timing of musical notes in your input. Use Rhythm assist when your melody contains the right notes but isn't consistently in time with the beat. Rhythm assist is available in Music studio.

### tempo

How fast music is played. Music typically follows a certain beat or meter, which drives the rhythm of the notes played. The speed of this beat is measured in beats per minute. A higher number of beats per minute corresponds to a faster tempo, or playback speed.

### Transformers

A generative AI technique that uses attention to understand the relationship between notes in a musical composition. To train the model, the musical data are converted into tokens. These tokens are used to represent a musical event in a given piece of music. During inference, the Transformers technique will extend your input track by up 30 seconds.

### TransformerXLClassical

A model based on the Transformer architecture. Compared with the traditional Transformer architecture, this model can better understand long-term dependencies and has decreased latency times.

### U-Net

A generative adversarial network (GAN) architecture built originally for image recognition tasks. The U-Net is a CNN. It's named for its U-like shape, in which the layers on the left side can pass information to the layers on the right side without passing through the entire neural network.

### update ratio

A hyperparameter that controls the number of model weight updates to the discriminator per update to the generator. A lower update ratio makes a stronger discriminator that can provide more accurate and useful information to the generator. The lower update ratio, however, increases training time.

### virtual keyboard

Also known as the AWS DeepComposer console-based keyboard. The virtual keyboard is available in the AWS DeepComposer Music studio. The virtual keyboard runs on any computer or mobile device that is connected to the AWS Cloud. You can use the virtual keyboard to play and record a short melody. You can then use the recorded melody with a supported AWS DeepComposer generative AI technique to create new musical compositions.

# AWS DeepComposer keyboard

The AWS DeepComposer keyboard is a music keyboard for learning generative AI and machine learning (ML), connect it to any computer that has access to the AWS DeepComposer console, play and record a short melody, and feed the melody to a supported generative AI architecture supported by AWS DeepComposer.

### Setup

1. Open the link your keyboard section of the AWS DeepComposer console.
2. Use the included USB cable to connect the keyboard to your computer.
3. On the back of the keyboard, locate the 8- or 16-digit alphanumeric serial number (S/N or DSN).
4. In **step 2** in the AWS DeepComposer console, enter the alphanumeric serial number.
5. Use Chrome browser.
6. keyboard features

<img src="https://github.com/og121public/aws-deepcomposer/blob/main/images/image1.jpg?raw=true" height="600">

The AWS DeepComposer keyboard is 32-key, 2 octave keyboard. It has several built-in features that are designed to increase the usability of the AWS DeepComposer keyboard.

### Arpeggiator (arp)

To create an arpeggio, use this button to step through a group of notes in a particular sequence to create an arpeggio.

### Auto-chord (chord)

Use this button to generate a chord by playing a single note. When you play a note, the auto-chord feature generates a simple triad chord using the note provided as the root note.

### Modulation

Use this wheel to modulate the synthetic audio signal to create a vibrato effect.

### Octave adjust

The default starting position of the AWS DeepComposer keyboard is the C4 and C3 octaves. The octave down button adjusts the keyboard down one or more octaves. The octave up button adjusts the keyboard up one or more octaves.

### Pitch

Use this wheel to slightly bend (change) the pitch of a note up or down.

### Playback

Use this button to playback a melody that you just recoreded.

### Record

Use this button to begin recording a melody on your keyboard. To stop recording, tap the button again.

### Sustain switch interface

You can plug any universal MIDI-compliant sustain pedal into your keyboard through this port.

### Tap tempo

Tap this button to set the arpeggio tempo manually.

### USB cable port

Use this port to connect the physical keyboard to the AWS DeepComposer console (Music studio).

### Volume

Use this slider enables to adjust the volume on your keyboard.

# How AWS DeepComposer works

At a fundamental level, music composition corresponds to sequences of notes of intricate tempo and dynamics. Musical compositions are categorized into genres based on discernible differences in the distribution different musical elements. To compose music for specific genre you have learn that genre specific distribution. At its core that is what the different generative AI techniques in AWS DeepComposer allow you to do.

This process involves figuring out the appropriate patterns and how to arrange them together in cohesive way. Aspiring composers typically have to spend several years training to learn these patterns. When composers graduate, they have developed an acute appreciation of the distributions of different musical elements in the genres they studied. Their understanding might be built on non-quantitative terms but they have built a quantitative distribution into their brains. They have developed a a natural neural network model of music composition and mastered ways to apply the model with great proficiency.

In machine learning, it's analogous to teaching a machine to compose music of a given genre. The machine learns the compositional features or patterns from a known musical collection to develop a practical understanding of the distribution of musical elements. The knowledge is built into an artificial neural network and represented by a set of optimal network weights of a chosen architecture.
