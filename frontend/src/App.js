import React, { useState } from 'react';
import {
  Container,
  Paper,
  Typography,
  TextField,
  Slider,
  Button,
  Box,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Grid,
} from '@mui/material';
import axios from 'axios';

function App() {
  const [numInputs, setNumInputs] = useState(2);
  const [inputs, setInputs] = useState([0, 0]);
  const [weights, setWeights] = useState([0.5, 0.5]);
  const [activation, setActivation] = useState('step');
  const [result, setResult] = useState(null);

  const handleInputChange = (index, value) => {
    const newInputs = [...inputs];
    newInputs[index] = parseFloat(value);
    setInputs(newInputs);
  };

  const handleWeightChange = (index, value) => {
    const newWeights = [...weights];
    newWeights[index] = value;
    setWeights(newWeights);
  };

  const handleNumInputsChange = (event) => {
    const newNum = parseInt(event.target.value);
    setNumInputs(newNum);
    setInputs(Array(newNum).fill(0));
    setWeights(Array(newNum).fill(0.5));
  };

  const calculateOutput = async () => {
    try {
      const response = await axios.post('http://localhost:5000/api/calculate', {
        inputs,
        weights,
        activation,
      });
      setResult(response.data);
    } catch (error) {
      console.error('Error calculating output:', error);
    }
  };

  return (
    <Container maxWidth="md" sx={{ py: 4 }}>
      <Typography variant="h3" component="h1" gutterBottom align="center">
        Perceptron Visualizer
      </Typography>
      
      <Paper elevation={3} sx={{ p: 3, mb: 3 }}>
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <TextField
              label="Number of Inputs"
              type="number"
              value={numInputs}
              onChange={handleNumInputsChange}
              inputProps={{ min: 1, max: 10 }}
              fullWidth
            />
          </Grid>

          <Grid item xs={12}>
            <FormControl fullWidth>
              <InputLabel>Activation Function</InputLabel>
              <Select
                value={activation}
                onChange={(e) => setActivation(e.target.value)}
              >
                <MenuItem value="step">Step Function</MenuItem>
                <MenuItem value="sigmoid">Sigmoid</MenuItem>
              </Select>
            </FormControl>
          </Grid>

          {Array.from({ length: numInputs }).map((_, index) => (
            <Grid item xs={12} key={index}>
              <Typography variant="subtitle1">Input {index + 1}</Typography>
              <Grid container spacing={2}>
                <Grid item xs={6}>
                  <TextField
                    label="Value"
                    type="number"
                    value={inputs[index]}
                    onChange={(e) => handleInputChange(index, e.target.value)}
                    fullWidth
                  />
                </Grid>
                <Grid item xs={6}>
                  <Typography gutterBottom>Weight: {weights[index].toFixed(2)}</Typography>
                  <Slider
                    value={weights[index]}
                    onChange={(_, value) => handleWeightChange(index, value)}
                    min={-1}
                    max={1}
                    step={0.01}
                  />
                </Grid>
              </Grid>
            </Grid>
          ))}

          <Grid item xs={12}>
            <Button
              variant="contained"
              color="primary"
              onClick={calculateOutput}
              fullWidth
            >
              Calculate Output
            </Button>
          </Grid>

          {result && (
            <Grid item xs={12}>
              <Paper elevation={2} sx={{ p: 2, bgcolor: 'background.default' }}>
                <Typography variant="h6">Results:</Typography>
                <Typography>Weighted Sum: {result.weighted_sum.toFixed(4)}</Typography>
                <Typography>Output: {result.output.toFixed(4)}</Typography>
              </Paper>
            </Grid>
          )}
        </Grid>
      </Paper>
    </Container>
  );
}

export default App; 