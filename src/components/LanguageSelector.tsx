'use client';

import React, { useState } from 'react';
import { Globe, Check } from 'lucide-react';

interface Language {
  code: string;
  name: string;
  nativeName: string;
  flag: string;
}

const SUPPORTED_LANGUAGES: Language[] = [
  // English variants
  { code: 'en-US', name: 'English (United States)', nativeName: 'English (US)', flag: '🇺🇸' },
  { code: 'en-GB', name: 'English (United Kingdom)', nativeName: 'English (UK)', flag: '🇬🇧' },
  { code: 'en-AU', name: 'English (Australia)', nativeName: 'English (AU)', flag: '🇦🇺' },
  { code: 'en-IN', name: 'English (India)', nativeName: 'English (IN)', flag: '🇮🇳' },
  
  // Indian languages
  { code: 'hi-IN', name: 'Hindi (India)', nativeName: 'हिन्दी', flag: '🇮🇳' },
  { code: 'ta-IN', name: 'Tamil (India)', nativeName: 'தமிழ்', flag: '🇮🇳' },
  { code: 'ml-IN', name: 'Malayalam (India)', nativeName: 'മലയാളം', flag: '🇮🇳' },
  { code: 'te-IN', name: 'Telugu (India)', nativeName: 'తెలుగు', flag: '🇮🇳' },
  { code: 'kn-IN', name: 'Kannada (India)', nativeName: 'ಕನ್ನಡ', flag: '🇮🇳' },
  { code: 'bn-IN', name: 'Bengali (India)', nativeName: 'বাংলা', flag: '🇮🇳' },
  { code: 'gu-IN', name: 'Gujarati (India)', nativeName: 'ગુજરાતી', flag: '🇮🇳' },
  { code: 'mr-IN', name: 'Marathi (India)', nativeName: 'मराठी', flag: '🇮🇳' },
  { code: 'pa-IN', name: 'Punjabi (India)', nativeName: 'ਪੰਜਾਬੀ', flag: '🇮🇳' },
  
  // European languages
  { code: 'es-ES', name: 'Spanish (Spain)', nativeName: 'Español', flag: '🇪🇸' },
  { code: 'es-MX', name: 'Spanish (Mexico)', nativeName: 'Español (MX)', flag: '🇲🇽' },
  { code: 'fr-FR', name: 'French (France)', nativeName: 'Français', flag: '🇫🇷' },
  { code: 'de-DE', name: 'German (Germany)', nativeName: 'Deutsch', flag: '🇩🇪' },
  { code: 'it-IT', name: 'Italian (Italy)', nativeName: 'Italiano', flag: '🇮🇹' },
  { code: 'pt-BR', name: 'Portuguese (Brazil)', nativeName: 'Português (BR)', flag: '🇧🇷' },
  { code: 'pt-PT', name: 'Portuguese (Portugal)', nativeName: 'Português (PT)', flag: '🇵🇹' },
  { code: 'ru-RU', name: 'Russian (Russia)', nativeName: 'Русский', flag: '🇷🇺' },
  { code: 'nl-NL', name: 'Dutch (Netherlands)', nativeName: 'Nederlands', flag: '🇳🇱' },
  
  // Asian languages
  { code: 'zh-CN', name: 'Chinese (Simplified)', nativeName: '简体中文', flag: '🇨🇳' },
  { code: 'zh-TW', name: 'Chinese (Traditional)', nativeName: '繁體中文', flag: '🇹🇼' },
  { code: 'ja-JP', name: 'Japanese (Japan)', nativeName: '日本語', flag: '🇯🇵' },
  { code: 'ko-KR', name: 'Korean (South Korea)', nativeName: '한국어', flag: '🇰🇷' },
  { code: 'th-TH', name: 'Thai (Thailand)', nativeName: 'ไทย', flag: '🇹🇭' },
  { code: 'vi-VN', name: 'Vietnamese (Vietnam)', nativeName: 'Tiếng Việt', flag: '🇻🇳' },
  
  // Arabic and Middle Eastern
  { code: 'ar-SA', name: 'Arabic (Saudi Arabia)', nativeName: 'العربية', flag: '🇸🇦' },
  { code: 'he-IL', name: 'Hebrew (Israel)', nativeName: 'עברית', flag: '🇮🇱' },
  
  // Nordic languages
  { code: 'sv-SE', name: 'Swedish (Sweden)', nativeName: 'Svenska', flag: '🇸🇪' },
  { code: 'no-NO', name: 'Norwegian (Norway)', nativeName: 'Norsk', flag: '🇳🇴' },
  { code: 'da-DK', name: 'Danish (Denmark)', nativeName: 'Dansk', flag: '🇩🇰' },
  { code: 'fi-FI', name: 'Finnish (Finland)', nativeName: 'Suomi', flag: '🇫🇮' },
];

interface LanguageSelectorProps {
  selectedLanguage: string;
  onLanguageChangeAction: (languageCode: string) => void;
  disabled?: boolean;
}

export default function LanguageSelector({ 
  selectedLanguage, 
  onLanguageChangeAction, 
  disabled = false 
}: LanguageSelectorProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');

  const selectedLang = SUPPORTED_LANGUAGES.find(lang => lang.code === selectedLanguage) || SUPPORTED_LANGUAGES[0];

  // Filter languages based on search term
  const filteredLanguages = SUPPORTED_LANGUAGES.filter(lang => 
    lang.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    lang.nativeName.toLowerCase().includes(searchTerm.toLowerCase()) ||
    lang.code.toLowerCase().includes(searchTerm.toLowerCase())
  );

  // Group languages by region
  const groupedLanguages = filteredLanguages.reduce((groups, lang) => {
    let region = 'Other';
    if (lang.code.includes('-US') || lang.code.includes('-GB') || lang.code.includes('-AU')) region = 'English';
    else if (lang.code.includes('-IN')) region = 'Indian Languages';
    else if (lang.code.includes('-ES') || lang.code.includes('-MX') || lang.code.includes('-FR') || 
             lang.code.includes('-DE') || lang.code.includes('-IT') || lang.code.includes('-PT') || 
             lang.code.includes('-RU') || lang.code.includes('-NL')) region = 'European';
    else if (lang.code.includes('-CN') || lang.code.includes('-TW') || lang.code.includes('-JP') || 
             lang.code.includes('-KR') || lang.code.includes('-TH') || lang.code.includes('-VN')) region = 'Asian';
    else if (lang.code.includes('-SA') || lang.code.includes('-IL')) region = 'Middle Eastern';
    else if (lang.code.includes('-SE') || lang.code.includes('-NO') || lang.code.includes('-DK') || 
             lang.code.includes('-FI')) region = 'Nordic';

    if (!groups[region]) groups[region] = [];
    groups[region].push(lang);
    return groups;
  }, {} as { [region: string]: Language[] });

  const handleLanguageSelect = (languageCode: string) => {
    onLanguageChangeAction(languageCode);
    setIsOpen(false);
    setSearchTerm('');
  };

  return (
    <div className="relative">
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        disabled={disabled}
        className={`flex items-center space-x-2 px-4 py-2 rounded-lg border font-medium transition-colors ${
          disabled 
            ? 'bg-gray-100 text-gray-400 cursor-not-allowed' 
            : 'bg-white hover:bg-gray-50 text-gray-700 border-gray-300'
        }`}
      >
        <Globe className="w-4 h-4" />
        <span className="text-lg">{selectedLang.flag}</span>
        <span className="hidden sm:inline">{selectedLang.nativeName}</span>
        <span className="sm:hidden">{selectedLang.code}</span>
        <svg className="w-4 h-4 ml-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {isOpen && !disabled && (
        <div className="absolute top-full left-0 mt-1 w-80 bg-white border border-gray-300 rounded-lg shadow-lg z-50 max-h-96 overflow-hidden">
          {/* Search input */}
          <div className="p-3 border-b border-gray-200">
            <input
              type="text"
              placeholder="Search languages..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              autoFocus
            />
          </div>
          
          {/* Language groups */}
          <div className="overflow-y-auto max-h-80">
            {Object.entries(groupedLanguages).map(([region, languages]) => (
              <div key={region}>
                <div className="px-3 py-2 bg-gray-50 border-b border-gray-200">
                  <span className="text-xs font-semibold text-gray-600 uppercase tracking-wide">
                    {region}
                  </span>
                </div>
                <div className="p-1">
                  {languages.map((language) => (
                    <button
                      key={language.code}
                      onClick={() => handleLanguageSelect(language.code)}
                      className={`w-full flex items-center space-x-3 px-2 py-2 rounded-md text-left hover:bg-gray-100 transition-colors ${
                        selectedLanguage === language.code ? 'bg-blue-50 text-blue-700' : 'text-gray-700'
                      }`}
                    >
                      <span className="text-lg">{language.flag}</span>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center space-x-2">
                          <span className="font-medium">{language.nativeName}</span>
                          {selectedLanguage === language.code && (
                            <Check className="w-4 h-4 text-blue-600" />
                          )}
                        </div>
                        <div className="text-sm text-gray-500">{language.name}</div>
                      </div>
                    </button>
                  ))}
                </div>
              </div>
            ))}
            
            {filteredLanguages.length === 0 && (
              <div className="p-4 text-center text-gray-500">
                No languages found matching &quot;{searchTerm}&quot;
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
