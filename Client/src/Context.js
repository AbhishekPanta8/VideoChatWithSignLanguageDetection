import React, { createContext, useState, useRef, useEffect } from 'react';
import io from 'socket.io-client';
import Peer from 'simple-peer';

const SocketContext = createContext();

const socket = io('http://127.0.0.1:5000');

const ContextProvider = ({ children }) => {
  const [callAccepted, setCallAccepted] = useState(false);
  const [callEnded, setCallEnded] = useState(false);
  const [stream, setStream] = useState();
  const [name, setName] = useState('');
  const [call, setCall] = useState({});
  const [me, setMe] = useState('');
  const [prediction, setPrediction] = useState('');

  const myVideo = useRef();
  const userVideo = useRef();
  const connectionRef = useRef();

  const sendFrames = () => {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    if (userVideo.current) {
      const { videoWidth, videoHeight } = userVideo.current;

      canvas.width = videoWidth;
      canvas.height = videoHeight;

      ctx.drawImage(userVideo.current, 0, 0, videoWidth, videoHeight);
    } else {
      const { videoWidth, videoHeight } = myVideo.current;

      canvas.width = videoWidth;
      canvas.height = videoHeight;

      ctx.drawImage(myVideo.current, 0, 0, videoWidth, videoHeight);
    }
    const frame = canvas.toDataURL('image/jpeg', 0.5);

    socket.emit('predictionVideo', frame);
  };

  useEffect(() => {
    navigator.mediaDevices.getUserMedia({ video: true, audio: true })
      .then((currentStream) => {
        setStream(currentStream);

        myVideo.current.srcObject = currentStream;
        myVideo.current.addEventListener('play', () => {
          setInterval(sendFrames, 2000);
        });
      });
  }, []);

  useEffect(() => {
    socket.on('me', (id) => setMe(id));

    socket.on('callUser', ({ from, name: callerName, signal }) => {
      setCall({ isReceivingCall: true, from, name: callerName, signal });
    });
    // Clean up the socket event listeners when the component unmounts
    return () => {
      socket.off('me');
      socket.off('callUser');
    };
  }, []);

  useEffect(() => {
    socket.on('predictionVideo', (predictedVal) => {
      setPrediction(predictedVal);
    });
    return () => {
      socket.off('predictionVideo');
    };
  }, [prediction]);

  const answerCall = () => {
    setCallAccepted(true);

    const peer = new Peer({ initiator: false, trickle: false, stream });

    peer.on('signal', (data) => {
      socket.emit('answerCall', { signal: data, to: call.from });
    });

    peer.on('stream', (currentStream) => {
      userVideo.current.srcObject = currentStream;
    });

    peer.signal(call.signal);

    connectionRef.current = peer;
  };

  const callUser = (id) => {
    const peer = new Peer({ initiator: true, trickle: false, stream });

    peer.on('signal', (data) => {
      socket.emit('callUser', { userToCall: id, signalData: data, from: me, name });
    });

    peer.on('stream', (currentStream) => {
      userVideo.current.srcObject = currentStream;
    });

    socket.on('callAccepted', (signal) => {
      setCallAccepted(true);

      peer.signal(signal);
    });

    connectionRef.current = peer;
  };

  const leaveCall = () => {
    setCallEnded(true);

    connectionRef.current.destroy();

    window.location.reload();
  };

  // console.log(myVideo);
  return (
    <SocketContext.Provider value={{
      socket,
      call,
      callAccepted,
      myVideo,
      userVideo,
      stream,
      name,
      setName,
      callEnded,
      me,
      callUser,
      leaveCall,
      answerCall,
      prediction,
    }}
    >
      {children}
    </SocketContext.Provider>
  );
};

export { ContextProvider, SocketContext };
