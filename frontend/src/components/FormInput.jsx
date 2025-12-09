import React from 'react';
import { TextField, MenuItem, FormControl, InputLabel, Select } from '@mui/material';

const FormInput = ({ label, type = 'number', value, onChange, name, options = null, step = 1, min, max }) => {
  if (options) {
    return (
      <FormControl fullWidth>
        <InputLabel>{label}</InputLabel>
        <Select
          value={value}
          label={label}
          onChange={(e) => onChange(name, e.target.value)}
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

  return (
    <TextField
      fullWidth
      label={label}
      type={type}
      value={value}
      onChange={(e) => onChange(name, type === 'number' ? Number(e.target.value) : e.target.value)}
      inputProps={{ step, min, max }}
      variant="outlined"
    />
  );
};

export default FormInput;