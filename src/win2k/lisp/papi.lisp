;; Using PAPI on a windows/intel machine from Allegro Common Lisp
;; from Franz Inc.  (see www.franz.com)
;; written by Richard Fateman, Univ. Calif, Berkeley 6/2002
;; no warranty, use at your own risk.

;; load winpapi.dll from its standard place.

(load "c:/winnt/system32/winpapi.dll")

(defparameter papi_values (make-array 80 :element-type '(unsigned-byte 8) :initial-element 0))
;; oddly enough the only "interface" problem was that Lisp didn't have
;; a LONGLONG.  So I wrote this program to convert an array of
;; 8 bytes-long objects into a corresponding array of
;; "arbitrary precision integers" or bignums.  

(defun bytes2ll(ba) ;; bytes to long long.  actually a bignum
  ;; 8 bytes represent a long-long. convert to bignum
  ;; ba is a byte array of length 8*n
  ;; return an array of n bignums.
  ;;
  (let* ((len (truncate (length ba) 8))
	 (a (make-array len :initial-element 0))
	 (i8 0)
	 (ans 0))
    (dotimes (i len a) ;return array a
      (setf i8 (1- (* (1+ i) 8)) ans 0)
      (dotimes (j 8 (setf (aref a i) ans))
	(setf ans (+(aref ba (- i8 j))(* 256 ans)))))))
      
;; There are lots of ways of using PAPI, but I only needed these
;; three calls. More parts of the connection could be added.

(ff:def-foreign-call
    (papi_start_counters "PAPI_start_counters")
    ((flags (* :int)) (len :int)) 
  :returning :int)

(ff:def-foreign-call
    (papi_read_counters "PAPI_read_counters")
    ((counters (* :int)) (len :int)) 
  :returning :int)

(ff:def-foreign-call
    (papi_stop_counters "PAPI_stop_counters")
    ((counters (* :int)) (len :int)) 
  :returning :int)

;; all negative numbers are errors.  see papi.h for decoding:
#|
#define PAPI_OK        0  /*No error*/
#define PAPI_EINVAL   -1  /*Invalid argument*/
#define PAPI_ENOMEM   -2  /*Insufficient memory*/
#define PAPI_ESYS     -3  /*A System/C library call failed, please check errno*/
#define PAPI_ESBSTR   -4  /*Substrate returned an error, 
			    usually the result of an unimplemented feature*/
#define PAPI_ECLOST   -5  /*Access to the counters was lost or interrupted*/
#define PAPI_EBUG     -6  /*Internal error, please send mail to the developers*/
#define PAPI_ENOEVNT  -7  /*Hardware Event does not exist*/
#define PAPI_ECNFLCT  -8  /*Hardware Event exists, but cannot be counted 
                            due to counter resource limitations*/ 
#define PAPI_ENOTRUN  -9  /*No Events or EventSets are currently counting*/
#define PAPI_EISRUN  -10  /*Events or EventSets are currently counting */
#define PAPI_ENOEVST -11  /* No EventSet Available */
#define PAPI_ENOTPRESET -12 /* Not a Preset Event in argument */
#define PAPI_ENOCNTR -13 /* Hardware does not support counters */
#define PAPI_EMISC   -14 /* No clue as to what this error code means */

|#


#| There are many possible events we can look for. Here are a few we
think are important for our testing, but (depending on your machine)
there are probably more. These numbers are from papiStdEventDefs.h
In our experience the fact that these items are defined does not
mean that they are actually implemented on your particular chip.
  
#define PAPI_L1_TCM  0x80000006 /*Level 1 total cache misses*/
#define PAPI_L2_TCM  0x80000007 /*Level 2 total cache misses*/
|#
(defconstant papi_l1_dcm  #x80000000)	;data cache misses, level 1
(defconstant papi_l1_icm  #x80000001)	;instruction cache misses
(defconstant papi_l1_tcm  #x80000006)   ;total cache misses
(defconstant papi_l2_tcm  #x80000007)
(defconstant papi_offset  #x80000000)

;;Live events on my pentium 3 are supposedly these on my pentium 3.
; I don't believe them though :
#| (for actual hex numbers, add papi_offset..)
  0	l1 cache
  1 	l1 inst
  6 	l1 total miss
  7 	l2 total miss
  a 	shr request for shared cache line
  b 	cln
  c 	inv
  d 	itv
 15 	instr trans lookaside buffer miss
 17 	l1 load miss
 18 	l1 store miss
 1b 	btac miss
 29 	hardware interrupts
 2b 	conditional branches executed
 2c 	conditional branches taken
 2d 	not taken
 2e 	mispredicted
 2f 	corrected predicted
 31 	total instr executed
 32 	integer inst executed
 34 	fp executed
 37 	total branch inst executed
 38 	vector / simd inst executed
 39 	fp inst per sec
 3a 	cycles process is stalled
 3c 	total cycles
 3d 	instr/sec
 40 	l1 d cache hit
 41 	l2 d cache hit
 42 	l1 d cache access
 43 	l2 d cache access
 46 	l2 d cache read
 49 	l2 d cache write
 4b 	l1 i cache hits
 4e 	l1 i cache accesses
 4f 	l2 i cache accesses
 51 	l1 i cache reads
 52 	l2 i cache reads
 54 	l1 i cache writes
 5a 	l1 total cache accesses
 5b 	l2 total cache accesses
 5e 	l2 total cache reads
 61 	l2 total cache writes
 |#


(defparameter *last-time* 0)

(defun start-ccm(); count L1 L2 total cache misses
  (let((ar2 
	(make-array 2 :element-type '(unsigned-byte 32) 
		    :initial-contents  (vector papi_l1_tcm papi_l2_tcm)))
       )
    (setf *last-time* (get-internal-run-time))
  (papi_start_counters ar2 2)
  ))

(defun read-ccm(printp)			;
  (papi_read_counters papi_values 2)
  (let* ((ans(bytes2ll papi_values))
	 (newtime (get-internal-run-time))
	 (diff (- newtime *last-time*)) )
    (setf *last-time* newtime)
    (if printp  (format t "~% L1 cache misses = ~e, L2 cache misses=~e, runtime=~s"
			(float (aref ans 0))
			(float (aref ans 1))
			diff)
      (list 	(aref ans 0)
		(aref ans 1)
		diff))))


;;; HERE ARE SOME OTHER EXAMPLES...

(defun start-ccmX(); count L1 data  and instruction cache misses
  (let(( ar2 
	 (make-array 2 :element-type '(unsigned-byte 32) 
		     :initial-contents  (vector papi_l1_dcm papi_l1_icm))))
  (papi_start_counters ar2 2)
  ))

(defun start-ccmY(); count inst and L1 total cache access
  (let(( ar2 
	 (make-array 2 :element-type '(unsigned-byte 32) 
		     :initial-contents  (vector (+ papi_offset #x31 )(+ papi_offset #x5a )))))
    (papi_start_counters ar2 2)))



(defun read-ccmX(printp)		;
  (papi_read_counters papi_values 2)
  (let ((ans(bytes2ll papi_values)))
    (if printp  (format t "~% L1 data misses= ~e, L1 inst misses=~e, total misses ~e" (float (aref ans 0))
			(float (aref ans 1))
			(float (+(aref ans 0)(aref ans 1)))))))

(defun read-ccmY(printp)		;
  (papi_read_counters papi_values 2)
  (let ((ans(bytes2ll papi_values)))
    (if printp  (format t "~% total instr = ~e, l1 cache accesses=~e" (float (aref ans 0))
			(float (aref ans 1))))))

(defun sc()(start-ccm)) ;; shorthand
(defun rc(p)(read-ccm p))		;p is t if you want to read and PRINT results

;; typical usage
;; (sc) ;start counting. e.g. L1 and L2 cache misses

;; (progn
;;    (rc nil)				;read and reset counters
;;    (compute something)
;;    (rc t) ;; print the results
;;   )


;;;;;;;;;;;;;;;;;;;;;;that's it;;;;;;;;;;;;;;;;;;;

