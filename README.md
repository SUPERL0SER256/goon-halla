GOONHALL- aka the gesture counter is a website that i made for avid gooners looking to improve their technique.
Thus i used Mediapipe( an open source framework for building Ml pipelines form processing real time data like video and audio in real time)
The user has to move their hand up an down as one does, and the website will track how many times the user does a stroke in a minute. 
the gesture has to be preciuse and fluid to count, thus improving ones technique as they play.
the main way of processing it is through the detecting the movement of the wrist as it is more stable than fingers and allows for variable grip postions.
Through opencv(cv2) we are able to process video from the webcam.
One up and down movement increases the count by 1 and this process is repeated until time runs out. 
Enjoy...
