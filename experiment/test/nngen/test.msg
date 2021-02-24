Call close ( ) instead of deactivate ( ) in CursorToBulkCursorAdaptor . close ( ) 
Removed non - needed imports 
Put that coffee down . 
fix error in docking station on chunk unload , fixes # 2898 
Fix bug 558 , PImage . save ( ) method not working with get ( ) 
Changed the behaviour of sendKey so that any key event kan be sent 
Ninja - add debug log statement to mv builder scheduled at startup 
Cancel all pending root scan requests on dispose . 
LPS - 48807 Remove EOL 
remove logall since it doesn ' t trigger anymore 
adds TODO for instantiateCroutonView 
annotate stacktrace only in " analyze stacktrace " and thread dump console . 
Fixed problem with getHighestFnScope method 
set component configurations correctly for trident spouts 
LPS - 50604 add @ Override 
Fixed bug in getType 
add " \ n " to processed response 
Add missing SideOnly in BaseMod 
fixed data info merge issue . 
Remove stray import of java . util . List 
Fix ringback return value . 
Suppress another flaky test . 
allow panes to raise themselves again 
Added missing method to check if a BaseObject3D . java is visible . 
set origin type with normal type 
Don ' t draw the current state if we are forcing another invalidation . 
Add @ Override to overriden method . 
Fix typo which cause stack overflow if children ( ) was called 
Fixes 
destroy folding model in three - side viewer 
IDEA - 103375 Fix understanding Git submodule roots 
Commit actually correct VizModel fix . 
Temporarily disable JavaLineIndentProvider not to break tests 
convert GitVersionTest to junit 
Add RIL_REQUEST_SET_INITIAL_ATTACH_APN . 
Tweak class loader 
Put back abstract version number 
Fix test failure on some platforms 
remove unused import 
Add field for java pc rendering 
Add isScreenOn ( ) to LocalPowerManager interface so we can call it from PhoneWindowManager 
Add warning when getContext is null . 
Fix setReleaseLabel ( ) . 
canvas . rotate ( ) expects degrees 
Deprecate SQLDialect . POSTGRESPLUS 
Fix NPE in getForegroundTintList 
Close the HTTP connection in the callable 
inadvertently removed method 
Remove extra spacing 
CLJ - 1854 make line number available on synthetic method 
Added parsing of Cython enums 
Changing default value for controlDir . 
Fixed problem with getHighestFnScope method 
don ' t need null check 
Remove accidental import 
Ignore WriteTimoutTestCase 
enable CAP_BLOCK_SUSPEND capability for system server 
Fix flag value collision 
remove " final " modifier for EventBus 
Add MEDIA_PROJECTION_SERVICE to the ServiceName list . 
close the OutputStream when audit has finished 
Added one line of space before main screen footer . 
Initial support for I18n ruby find usages support + fixed tests on yaml backend 
Allow external file picker 
NumberPicker truncates the edited value . 
Make gestures visible again . 
create configBase , since it is no longer created elsewhere . 
SONAR - 5329 - Added activity table to DatabaseVersion for cleaning . 
Remove dead variable _hdfsActive . 
Fix indent 
Set TWLayoutManager to have vertical orientation by default 
DO NOT MERGE Fix finalization 
Fix NPE in volley stream fetcher 
default native proto to correct port in Config 
hide AudioAttributes . usageToString ( ) method 
fix bug in non - choreographer looper 
IDEA - 95867 please map an " XML Catalog " topic 
Remove unnecessary getter 
Covariant Equals Check compilable UT inputs 
woops java type checks ! 
Make app module field private . 
Disable other test . 
Shutdown the actor system after we are done with it . 
Fixed bug in new entity sound hook . 
HHH - 9201 - Evict cache instead of remove after transaction completion 
Disable watch service based file watching for now 
Fix hashCode 
include separator below title 
Don ' t run dSYM on bundles 
Remove recycle of B2CConverter when used for URI conversion . 
Added toString ( ) method in NeoApp . NeoAppRelationshipType . . . otherwise 
Fix test 
Don ' t return an unmod collection . 
Fix AccountManagerServiceTest 
Add temporary class object icon for structure and project views 
Actually set the autobind flag with the specified value ( # 1038 ) . 
removed normalize from lfw 
Make the remote control desktop sharing check box non - opaque on Windows and Linux . 
Remove invalid generics from warningsOverload group . 
Replace hardcoded PI with Math . PI 
fixed multiple db calls 
LPS - 67643 reactivate test 
Add options . setTrustedStrings ( true ) to ant jscomp 
 @ Deprecated does not have effect in Groovy 
delete obsolete TODO 
Remove unused field from AssetManager . 
Removed unnecessary tag 
Fix a broken test utility causing many Apple tests to be skipped 
Fix build break 
Consider default zoom scale for mobile viewport . 
Fixes java 1 . 5 compatibility . 
Remove unnecessary method 
LPS - 57379 Fix potential future bug 
add module settings action to dependencies trees 
Make PhoneAccount . CAPABILITY_VIDEO_CALLING public 
move test file into temp folder 
Fix missing Date import in Restore . java 
Use this ( ) in constructor 
add Gpu - Image library 
Fixing a failing unit test . 
Removing unused override . 
Fix an infinite exception storm triggered by DefaultChannelHandlerContext . invokeInboundBufferUpdated ( ) 
Javac ( at least for 1 . 7 . 0_51 on Mac ) wants fully qualified class name 
Removed a TODO mark which is not going to be fixed ( it just works and is not reusable at this moment . ) 
added missing BenchmarkRunner config for JSON - lib JSON databind 
Catch badly formed URLs in WebView accessibility injector . 
Add GC thread to RSGL . 
Replacing correct XML tag on editing external annotations . 
Fix naming issue in comment 
Fix indentation 
Make final 
Make PyUnbound Local Variable inspection enabled by default 
Fix bug 5248444 - ViewPager reported scroll state out of sync for some 
reset when release 
rewrite readonly status dialog ( IDEADEV - 2453 , IDEADEV - 8703 , IDEADEV - 12772 ) 
Correct compile error . 
Do not show article tab for non - articles 
update the conversations view ( and the lock icon ) after receiving device list 
fixed compilation error 
Fix leaked broadcast receiver 
Fixed action mode bug with select show on map in my tracks 
PY - 1065 Reformat and auto - import do not respect space - after - comma preference in multiline imports 
help ID ( IDEA - 57112 ) 
changed return type to be more generic 
Fixed the animation end event . 
Fixed checkstyle error 
Fix indent 
LPS - 48154 Expose it or else Eclipse complains 
Switch animator to fixed - rate scheduling 
Fix NPE in BrowseFragment header transition before API 19 
DomModelTreeView disposing fixed 
add method GraphTableCell . getRefsToThisCommit ( ) 
fixed javadoc 
Shutdown of OSoftThread now also interrupts the associated thread 
LPS - 53947 add @ Override 
Added comment to eagerly initialized singleton implementation . 
Allow categorical encoding . 
Add an ensureOpen check to Publication . positionLimit ( ) . 
Missing first line of license in header . 
Fix warnings 
Remove redundant test 
Should not be closing system dialogs when Recents is hidden . 
Fix missing imports . 
hide structure by default 
shutting off external stop messages once again 
invalidate WebView after Title change 
cherry pick " Fix build " a227db53f0cd30214c393b400436d1e0863396ef into master 
Add Javadoc 
remove me note 
hide secondary toolbar on startup 
Added getter to the user object associated with a base adapter helper 
VolumePreference . onActivityStop only stops playback 
Initialize audio quality with NONE instead of AMR_WB . 
LPS - 33429 missing transactional annotation 
Rogue line number for try / catch statement confuses IDE debuggers 
Fix conditional . 
Exploding Engines should kill themselves 
Fix ApplicationInfo copy ctor 
Replaced calls to HttpServletRequest . getScheme ( ) , HttpServletRequest . host ( ) and HttpServletRequest . getPathInfo ( ) with a call to HttpServletRequest . getServletURL ( ) 
LPS - 38606 Remove unthrown exception 
add todo 
remove unused import 
Add access to ContentProvider for SyncState . 
Set mImeBackPressed to true when presenting the image settings dialog . 
help topic ( PY - 348 ) 
Fix NullPointerException 
fixed fragements back stack problem 
Identity . combine ( other ) should just return other . 
IDEA - 55132 Provide ability to setup formatter to use indents relative to expression start 
added type name . 
build # 3 
FastArray doesn ' t overwrite toString ( ) method 
Added type for iterator 
Suppress exception log 
Allow subclasses to fiddle with mSceneInitialized 
Fix NPE in the test code 
Removes unused SuppressWarnings 
Make LightweightMessageFormatter . withoutSource ( ) public . 
fixed previous check - in 
Send ACTION_DEVICE_POLICY_MANAGER_STATE_CHANGED Broadcast when a user restriction has been changed . 
Invalidate cached data for new name when renaming a table 
LPS - 39137 Fix bad SF compile error 
Add shadow for View . MeasureSpec class 
Fix default TextView gravity 
XML object now cross references loadXML ref docs 
Fix bug # 5544127 TextView regression in ICS when setting TextSize dynamically 
hide no - tabs toolwindow on middle - click on base label 
Revert " Decreasing default max connections per node to 10 " 
LPS - 55112 It ' s already static 
Remove empty destroy ( ) method from GLColorOverlay 
Fix typo 
Fix docs typo in YUV code 
Fix bug with bad state in animators 
Fix random text insertion 
GRAILS - 6047 - this change may resolve GRAILS - 6047 , pending further investigation 
Added invalidateMaster method to control that behaviour from outside 
refresh settingsModified flag on apply ( ) 
Upped initial buffer size to 1K to be consistent with the pooled version 
add missing branch 
removed two extra lines from the merge in r2055 
FredrichO / JonathanL - fixed issue with views not recycling on media grid 
Fix broken build . 
Removed unnecessary annotation 
Remove stray popBackStack Fixes # 3446 
made FnExpr public 
Fix AbstractTFS compilation with hdfs1 
Invalidate SubtitleView during setText ( ) 
Added redundant abstract method declarations to maybe work around problems on Nokias 
reset last rendered text on render end 
Fix sysout with trailing messages after a newline 
add @ RemotableViewMethod to TextView # setEnabled 
Removed unused import 
LPS - 53563 Plugin visibility in personal sites is impacted by the user not being in the Power User role 
Add missing access check . 
Adding print to err stream when there ' s an HTTP call made that doesn ' t match any response rules . 
Renamed an incorrectly named test . 
Fix typo 
Suppress NoClassDefFoundError ( support info ) 
Prevent new ripples when max reached 
Fix a problem with some hardware devices . 
reset CodeInsightTestFixture in tearDown 
Remove unneeded commented - out SuppressWarnings . 
don ' t sanitize pipeline group name in getGroup 
Removed system out . 
shutting down the listener thread properly { SMACK - 120 } 
Enable strict mode assumption for those using ecmascript 5 . 
Throw UnsupportedOperationException in not implemented method 
Check for correct capability for showing desktop sharing 
Fix a test failure after the timeout refactoring 
Avoid NPE on shutdown if application doesn ' t use WebSocket 
Reinstantiated ignored test . 
Add synchronization to close ( ) 
Fix bug 3497341 - take list padding into account when drawing 
Fix NPE when branding isn ' t present 
fixed # 1156 
Fixed small comments 
focus editor on Cmd + E find next 
Fix bug in ByteBuffer . getAll if the list is empty 
Fixed a bug where the margins where not updated correctly 
Accessibility focus should not affect the currently active window . 
Stop any running DHCP client before bringup up an Ethernet interface 
fixed compilation error for ' Html ' class usage . 
IDEA - 146663 Typing on the replace field causes search preview to refresh 
Add support for the Firebird database - Fixed CURRENT_USER ( ) 
Fix test ' s name 
Migrate UI to supportsMultipleUsers ( ) . 
give access to the defining classloader 
Revert " Fix Forge ore dictionary support for the Assembly Table " 
Add private constructor to tachyon . Format 
Properly remove references to the original CompilerInput after the hot - swap is done . 
Fix build 
Fix return node after addding other node . 
set origin type with normal type 
Hopefully fix # 48 
Remove an unneeded assignment 
Add indices . requests . cache . clean_interval as a known settings 
add correct exception throw declaration to isScreenBrightnessBoosted . 
FileSystemCompiler # generateFileNamesFromOptions is adding filenames multiple times 
remove javadoc warning 
Fixed soft - dependencies broken by previous commit . 
Fix Yahoo MUC . 
remove redundant code from # 2051 
Reduced visibility of AggregationOptions . Builder constructor to private 
don ' t update file status if project is already disposed ( IDEADEV - 25430 ) 
removed unused method 
Remove stray semi - colon . 
Build layer immediately when setting it in ViewPropertyAnimator 
specify default service name for coordinator 
Fix build . 
Initialize smartLockAutoFill again 
Fix off - by - one issue when removing from Iterator 
LPS - 50974 Fix compile error 
Add a blank line to MockRilTest . java 
DO NOT MERGE Use correct param for heightMode in RelativeLayout horizontal measure 
disable reloading for scripts with dates in the future 
Don ' t draw the selection pointer on long press . 
Fixed the check used to decide if it the given view id is from an action item or a simple view . 
typo cont ' d ( remove deprecated method in trunk ) 
LPS - 56976 CounterTransactionExecutor does not need the BaseTransactionExecutor , just directly implements TransactionExecutor 
removed assignment in PSurfaceJOGL 
update FileRequest 
Shorten the interval to reconnect the disconnected IME 
Kill a once - used temporary variable 
add @ Deprecated to make sure it is not part of 1 . 0 . 0 API 
Fix noisy exception logging 
Add break statement on load extractor from form 
LRQA - 29060 Never set result to NULL 
extracted InvocationOnMock interface 
Fix small rendering bug when item is hovering over AdvancedSlot . 
Schedule traversal instead performing it directly . 
Fix problem in mFileProjectionMap 
Added GL_PATCHES as new constant 
LPS - 48078 Add unit tests 
Fix SymbolExtractor implementation for UnionNode 
Changed min_adapter_stack to < = 
Fix FindBugs warning 
Fix NPE in style fragment parse 
Preserve explicit consumed state in replaceSystemWindowInsets DO NOT MERGE 
Make isms optional ( support info ) 
Remove focus from focused control when dialog is loaded 
Set default icon size to 24dp 
Missing tld . 
LPS - 18821 Add it to the interface so it can be called ( in case it ' s needed in the future ) 
Fix javadoc of DBCursor # curr ( ) 
Updated the package description 
IDEA - 73567 SVN repository browser doesn ' t work for default project any more 
Add missing commit 
Missed import 
Fix deadlock in XMPPConnection while login and parsing stream features ( for Tigase servers ) 
Remove code highlighted as unused by UCDetector 
remove debug message 
Remove useless code . 
reverted SafeVarargs because of Oracle license restrictions 
add filter and scope parameters when creating terms stats facet 
Fix build break 
Print " Success " on success 
Add missing @ Override annotation . 
Invalidate progress bar every time the holding cell is requested for uploads in progress 
fixed issue # 199 
removed override . 
Added support for tracking self hosted uniques . 
Restore default favicon when page loads so as to prevent a favicon from the last webpage being used 
disable optimizations for interfaces 
Fix a bug disappearing installerPackageName info of packages after reboot . 
Removed useless variables . 
Fix build due to merge of 7595842 and renaming due to 8eb87f0 
Fixed setter in RuntimeEnvironment ( lucene locking ) 
trim user defined string 
removed extraneous import 
add missing call to superclass method 
Resolve potential deadlock state during EsThreadPoolExecutor shutdown 
Fixed added whitespace in AbstractAtomicReferenceOperation . 
initialize codec list before MediaCodec . getCodecInfo ( ) 
Remove sysout 
Add @ Test annotation for outerjoin test 
Added ( back ) counter increment when sending packets . 
Added missing @ Override annotation in NearCacheStats . 
Adding pan behavior on preference dialog . 
Stack overflow fix 
added missing header 
Fix checkstyle 
Fix JDBC DatabaseMetaData . getIdentifierQuoteString 
remove an unused import 
Initialize country and dialect specific engines . 
Haptics for PIN enter key 
Improved error handling when font loading fails . 
Fix InputType determination of WebTextView 
Fixed wrong class filter condition 
Make an internal class package private " 
Fix type in TwitterRestClient . 
Fix comment 
Fixed cloning problem 
Fixed formatting / printing bug in CXDChunk . 
added finally block that sets the system property back to null after use so as not to break other tests 
Remove redundant ` private ` modifier from enum ctor 
remove MAX_TRANSFERS from RepeatedRaptorProfileRouter ( not used ) . 
Remove unneeded break statement . 
Revert " LPS - 37902 Deleting file entry types too " 
add a * 
removed unnecessary import 
Delete unused method 
Disabling all version removal for test 
Add missing @ Override annotation . 
LPS - 37741 Removed unused annotation 
removed unnecessary import 
DO NOT MERGE 2928156 Accessibility event ' s parcelableData property not nullified during recycling ( GINGERBREAD ) 
fixed bug in path builder , close # 1490 
Make getListView ( ) public and hidden , as it is needed for WifiSettings . 
Remove duplicate code 
Remove unused imports in Shape2D 
Removed @ Override annotation in JetContentBasedFileSubstitutor . getDecompiledPsiFile ( ) to avoid broken compilation . 
Changed power request to activation energy . 
LPS - 58126 Revert " LPS - 56015 We only want the regular parameters , see getParameterValues and other methods in the class . " 
remove forgoten repeat from test 
Fix null check for BaseObj . equals . 
Fixed inverse of listener assignment . 
Disable RenderThread 
Removed CompositeRunner constructor with just a class 
added call to canvas . setFocusTraversalKeysEnabled ( ) in initSurface ( ) to 
Fix start delay jank in ViewPropertyAnimator 
Fix crash when converting empty string to IOB datums 
Add AbstractProcessingFilter . getAuthenticationDetailsSource ( ) . 
fix missing processFormFields 
Fix condition to check whether or not the database needs upgrading 
fixed # 425 
Remove unnecessary code . 
Improve street name display 
Removed the testing code 
Align ssl test in parseHost in BIO and NIO & APR 
Fix bug 43236 . 
Now closes ReadableByteChannel after it has been read . 
Remove misplaced @ return in JavaDoc 
UNDERTOW - 904 Fix close issue with HTTP / 2 
add not null 
Fix Visibility transitions canceling when nothing has changed . 
Remove unused boolean parameter 
remove unused constant 
 ( If there is no error , do change the target 
Resolved conflict 
Re - add the proper accessor 
handle execptions while writting to the debug stream . 
Changed delay of the runnable in startSyncFolderOperation of FileDisplayActivity , adding 350ms 
make changelogParamters publicly accessable 
add stop id to json serialization ( reported by Colin ) 
setting backround color if image comes directly from cache 
make package protected , so other apps do not link it 
do not stop editing on isModified invokation 
PagedFileStorage $ StorageLock $ BuffersCache . createValue 
update shelf on project open ( IDEADEV - 12518 ) 
Fix memory leak 
