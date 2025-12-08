import React from 'react';
import { Box, Typography } from '@mui/material';
import { School } from '@mui/icons-material';

const Footer = () => {
  return (
    <Box
      component="footer"
      sx={{
        py: 3,
        px: 2,
        mt: 'auto',
        backgroundColor: 'white',
        borderTop: '1px solid',
        borderColor: 'divider',
        textAlign: 'center'
      }}
    >
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 1, mb: 1 }}>
        <School color="primary" />
        <Typography variant="body1" fontWeight="bold" color="text.primary">
          COMP-258 Neural Networks Project
        </Typography>
      </Box>
      <Typography variant="body2" color="text.secondary">
        Full-Stack Intelligent Application | {new Date().getFullYear()}
      </Typography>
    </Box>
  );
};

export default Footer;