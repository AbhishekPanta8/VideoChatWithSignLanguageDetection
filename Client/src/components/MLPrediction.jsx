import React, { useState, useEffect } from 'react';
import { Typography } from '@material-ui/core';

const MLPrediction = ({ prediction }) => {
  const [predictionArray, setPredictionArray] = useState([]);

  useEffect(() => {
    if (prediction) {
      setPredictionArray((prevArray) => [...prevArray, prediction]);
    }
  }, [prediction]);

  return (
    <>
      <br />
      Please wait initially for 30 seconds as we are feeding model for first click
      <div style={{ display: 'flex', alignItems: 'center' }}>
        <Typography variant="h6" gutterBottom>User1:</Typography>
        {predictionArray.map((currentPrediction, index) => (
          <Typography key={index} variant="h6" gutterBottom style={{ marginLeft: '10px' }}>
            {currentPrediction}
          </Typography>
        ))}
      </div>
    </>
  );
};

export default MLPrediction;
