BEGIN			{
			FS=":"
			current=""
			printf("/* automatically generated, do not edit */\n");
			}

$1 == "begin"		{
			current = $2;
			printf("\n");
			next;
			}

$1 == "include"		{
			next;
			}

$1 == "qualifier"	{
			next;
			}

$1 == ""		{
			next;
			}

$1 ~ "#.*"		{
			next;
			}

NF >= 2			{
			printf("#define %s_%s\t0x%s\n", current, $2, $1);
			next;
			}
