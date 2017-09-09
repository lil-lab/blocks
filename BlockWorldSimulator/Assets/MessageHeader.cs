using System;

public enum MessageHeader { SuccessACK, 			// Job succeeded
					 FailureACK, 			// Job Failed
					 SuccessAndResetACK,    // Job succeeded and reset to new episode
					 FailureAndResetACK,     // Job failed and reset to new episode
					 Reset					// resetting to new episode
				   };  
