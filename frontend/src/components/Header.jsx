import React from 'react';
import { AppBar, Toolbar, Typography, Chip, Box } from '@mui/material';
import { Psychology, CheckCircle, Error, HourglassEmpty } from '@mui/icons-material';

const Header = ({ apiStatus }) => {
  const getStatusColor = () => {
    switch (apiStatus) {
      case 'connected':
        return 'success';
      case 'disconnected':
        return 'error';
      default:
        return 'warning';
    }
  };

  const getStatusIcon = () => {
    switch (apiStatus) {
      case 'connected':
        return <CheckCircle />;
      case 'disconnected':
        return <Error />;
      default:
        return <HourglassEmpty />;
    }
  };

  return (
    <AppBar position="static" sx={{ 
      background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
      boxShadow: 3
    }}>
      <Toolbar sx={{ justifyContent: 'space-between', py: 1 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Psychology sx={{ fontSize: 40 }} />
          <Typography variant="h5" component="h1" fontWeight="bold">
            Neural Networks Student Analytics
          </Typography>
        </Box>
        
        <Chip
          icon={getStatusIcon()}
          label={`API: ${apiStatus}`}
          color={getStatusColor()}
          sx={{ 
            fontWeight: 'bold',
            textTransform: 'capitalize',
            px: 1
          }}
        />
      </Toolbar>
    </AppBar>
  );
};

export default Header;