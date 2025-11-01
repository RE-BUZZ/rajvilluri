'use client';
import React from 'react';
import AvatarVideo from '@/components/AvatarVideo';


export default function Home() {
  return (
    <div className="min-h-screen bg-white">
      <AvatarVideo 
        avatarId="Katya_CasualLook_public" 
        voiceId="95856005-0332-41b0-935f-352e296aa0df" 
      />
    </div>
  );
}