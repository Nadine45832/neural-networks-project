import React from 'react';
import { TextField, MenuItem, FormControl, InputLabel, Select } from '@mui/material';

const FormInput = ({ 
  label, 
  type = 'number', 
  value, 
  onChange, 
  name,
  options = null,
  step = 1,
  min,
  max
}) => {
  // If options are provided, render a select dropdown
  if (options) {
    return (
      <FormControl fullWidth>
        <InputLabel>{label}</InputLabel>
        <Select
          value={value}
          label={label}
          onChange={(e) => onChange(name, Number(e.target.value))}
        >
          {options.map((option) => (
            <MenuItem key={option.value} value={option.value}>
              {option.label}
            </MenuItem>
          ))}
        </Select>
      </FormControl>
    );
  }

  // Otherwise render a text field
  return (
    <TextField
      fullWidth
      label={label}
      type={type}
      value={value}
      onChange={(e) => onChange(name, Number(e.target.value))}
      inputProps={{ 
        step: step,
        min: min,
        max: max
      }}
      variant="outlined"
    />
  );
};

export default FormInput;