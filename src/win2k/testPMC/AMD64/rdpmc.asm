PUBLIC __rdpmc
.CODE
__rdpmc PROC NEAR
	rdpmc			; the pmc to read must be in eax
    ret				; it is returned in edx:eax
__rdpmc ENDP
END
