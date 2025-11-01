'use client';

import React, { useState } from 'react';
import { Settings, Save } from 'lucide-react';

interface AvatarSettingsProps {
  avatarId: string;
  voiceId: string;
  onSettingsChange: (avatarId: string, voiceId: string) => void;
}

export default function AvatarSettings({ 
  avatarId, 
  voiceId, 
  onSettingsChange 
}: AvatarSettingsProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [tempAvatarId, setTempAvatarId] = useState(avatarId);
  const [tempVoiceId, setTempVoiceId] = useState(voiceId);
  
  const handleSave = () => {
    onSettingsChange(tempAvatarId, tempVoiceId);
    setIsOpen(false);
  };
  
  const popularAvatars = [
    { id: 'Katya_ProfessionalLook2_public', name: 'Wayne - Professional Male' },
    { id: 'Susan_20240715', name: 'Susan - Professional Female' },
    { id: 'Tyler-incasualsuit-20220721', name: 'Tyler - Casual Male' },
    { id: 'Anna_public_3_20240108', name: 'Anna - Casual Female' },
    { id: 'josh_lite3_20230714', name: 'Josh - Business Male' },
    { id: 'Vanessa_public_2_20240108', name: 'Vanessa - Business Female' },
  ];
  
  const popularVoices = [
    { id: '077ab11b-f1b1-4d6d-8b17-c4e2e1a2c947', name: 'Ryan - American Male' },
    { id: '95856005-0332-41b0-935f-352e296aa0df', name: 'Emma - American Female' },
    { id: '38bc5bd7-f5dd-429c-8bb3-bb23c7c96e8e', name: 'William - British Male' },
    { id: 'b7395f42-8c3d-4a79-88b1-8c8b8a01d5c8', name: 'Charlotte - British Female' },
    { id: 'fa8d1d2b-e8c9-4c8f-b8a1-8f8b8b8b8b8b', name: 'David - Australian Male' },
    { id: '8b8b8b8b-8b8b-8b8b-8b8b-8b8b8b8b8b8b', name: 'Olivia - Canadian Female' },
  ];
  
  if (!isOpen) {
    return (
      <button
        onClick={() => setIsOpen(true)}
        className="fixed top-4 right-4 flex items-center space-x-2 bg-gray-800 hover:bg-gray-700 text-white px-4 py-2 rounded-lg shadow-lg z-10"
      >
        <Settings size={20} />
        <span>Settings</span>
      </button>
    );
  }
  
  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg p-6 w-full max-w-2xl max-h-[90vh] overflow-y-auto">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-2xl font-bold">Avatar Settings</h2>
          <button
            onClick={() => setIsOpen(false)}
            className="text-gray-500 hover:text-gray-700 text-xl font-bold"
          >
            ×
          </button>
        </div>
        
        <div className="space-y-6">
          {/* Avatar ID Section */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Avatar ID
            </label>
            <input
              type="text"
              value={tempAvatarId}
              onChange={(e) => setTempAvatarId(e.target.value)}
              placeholder="Enter avatar ID"
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
            
            <div className="mt-3">
              <p className="text-sm text-gray-600 mb-2">Popular Avatars:</p>
              <div className="grid grid-cols-1 gap-2">
                {popularAvatars.map((avatar) => (
                  <button
                    key={avatar.id}
                    onClick={() => setTempAvatarId(avatar.id)}
                    className={`text-left p-2 rounded border transition-colors ${
                      tempAvatarId === avatar.id
                        ? 'bg-blue-100 border-blue-500 text-blue-700'
                        : 'bg-gray-50 border-gray-200 hover:bg-gray-100'
                    }`}
                  >
                    <div className="font-medium">{avatar.name}</div>
                    <div className="text-xs text-gray-500">{avatar.id}</div>
                  </button>
                ))}
              </div>
            </div>
          </div>
          
          {/* Voice ID Section */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Voice ID
            </label>
            <input
              type="text"
              value={tempVoiceId}
              onChange={(e) => setTempVoiceId(e.target.value)}
              placeholder="Enter voice ID"
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
            
            <div className="mt-3">
              <p className="text-sm text-gray-600 mb-2">Popular Voices:</p>
              <div className="grid grid-cols-1 gap-2">
                {popularVoices.map((voice) => (
                  <button
                    key={voice.id}
                    onClick={() => setTempVoiceId(voice.id)}
                    className={`text-left p-2 rounded border transition-colors ${
                      tempVoiceId === voice.id
                        ? 'bg-blue-100 border-blue-500 text-blue-700'
                        : 'bg-gray-50 border-gray-200 hover:bg-gray-100'
                    }`}
                  >
                    <div className="font-medium">{voice.name}</div>
                    <div className="text-xs text-gray-500">{voice.id}</div>
                  </button>
                ))}
              </div>
            </div>
          </div>
          
          {/* Instructions */}
          <div className="bg-blue-50 p-4 rounded-lg">
            <h3 className="font-medium text-blue-900 mb-2">How to get Avatar and Voice IDs:</h3>
            <ul className="text-sm text-blue-800 space-y-1">
              <li>• Visit <a href="https://labs.heygen.com/interactive-avatar" target="_blank" rel="noopener noreferrer" className="underline">labs.heygen.com/interactive-avatar</a> to find Avatar IDs</li>
              <li>• Use the HeyGen API <a href="https://docs.heygen.com/reference/list-voices-v2" target="_blank" rel="noopener noreferrer" className="underline">List Voices endpoint</a> to get Voice IDs</li>
              <li>• You can create custom avatars at <a href="https://labs.heygen.com/interactive-avatar" target="_blank" rel="noopener noreferrer" className="underline">labs.heygen.com</a></li>
            </ul>
          </div>
        </div>
        
        <div className="flex justify-end space-x-3 mt-6">
          <button
            onClick={() => setIsOpen(false)}
            className="px-4 py-2 text-gray-600 bg-gray-200 hover:bg-gray-300 rounded-lg font-medium"
          >
            Cancel
          </button>
          <button
            onClick={handleSave}
            className="flex items-center space-x-2 px-4 py-2 bg-blue-500 hover:bg-blue-600 text-white rounded-lg font-medium"
          >
            <Save size={16} />
            <span>Save Settings</span>
          </button>
        </div>
      </div>
    </div>
  );
}