  void Fed9UDescriptionToXml::addStripsAttribute(DOMElement* childElement) throw (Fed9UXMLDescriptionException)
  {
    try {
      // now we must loop over all strips building all the info for each 
      // strip (pedestal, lowthresholdfactor, highthresholdfactor, noise, disable)
      // into one blob. To do this we use a stringstream streaming in values as hex
      // before that we have to limit the values to the correct precision and number of bits before constructing the binary buffer
      // each strip will contain the data in the following format:
      //
      // *****************************************
      // * Value       *   bits   *   range      *
      // *****************************************
      // * Pedestal    *   [31-22]*   0-1023     *
      // * Noise       *   [21-13]* 0.0-51.1(0.1)*
      // * High Factor *   [12-7] * 0.0-12.6(0.2)*
      // * Low Factor  *   [6-1]  * 0.0-12.6(0.2)*
      // * Disable     *   [0]    * 0,1          *
      // *****************************************
      //
      
      u32 stripData;
      u32 low = 0;
      u32 high = 0;
      u32 ped = 0;
      u32 noise = 0;
      const Fed9UStripDescription * theFed9UStripDescription; 
      ostringstream charData; 
      charData.str("");
      charData << std::hex << std::setfill('0') << std::setw(8);
      for (int i=0;i<STRIPS_PER_APV;i++) {
	stripData = 0;
	//Set the Fed9UAddress using the the stripId, and get the corresponding Fed9UStripDescription from Fed9UDescription.
	theFed9UAddress.setApvStrip(i);
	theFed9UStripDescription = &theFed9UDescription.getFedStrips().getStrip(theFed9UAddress);
	
	low = (static_cast<u32>(theFed9UStripDescription->getLowThresholdFactor()*5.0 + 0.5) ) & 0x3F; 
	high = (static_cast<u32>(theFed9UStripDescription->getHighThresholdFactor()*5.0 + 0.5) ) & 0x3F;
	noise = static_cast<u32>(theFed9UStripDescription->getNoise()*10.0 + 0.5) & 0x01FF;
	ped = static_cast<u32>(theFed9UStripDescription->getPedestal()) & 0x03FF;

	stripData = (ped << 22) | (noise << 13) | (high << 7) | (low << 1) | ( theFed9UStripDescription->getDisable() ? 0x1 : 0x0 );
	//	if( i == 0 ) {
	// std::cout << "the strip data is :" << endl;
	//std::cout << std::dec << "low = " << low << ", high= " << high << ", noise= " << noise << ", ped= " << ped << std::hex << std::setfill('0') << std::setw(8) << stripData << " " << endl;
	//}
	charData << std::hex << std::setfill('0') << std::setw(8) << stripData;
      }
      childElement->setAttribute(X("data"), X(charData.str().c_str()));
    } 
    //.......
  }
